import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import SimpleITK as sitk


from slicedMRI.dataset import PairedMRIDataset
from slicedMRI.config import DatasetConfig


def get_data_dicts(args: DatasetConfig):
    lr_dir: Path = args.data_dir / args.lr_base
    hr_dir: Path = args.data_dir / args.hr_base
    data_dicts = []

    for subject_dir in hr_dir.glob("sub-*"):
        subject_id = subject_dir.name
        hr_files = list((subject_dir / "anat").glob(f"*{args.modality}*.nii*"))
        lr_subject_dir = lr_dir / subject_id
        if not lr_subject_dir.exists():
            continue
        lr_files = list((lr_subject_dir / "anat").glob(f"*{args.modality}*.nii*"))
        if hr_files and lr_files:
            hr_path = str(hr_files[0])
            lr_path = str(lr_files[0])
            prompt = f"high quality MRI scan, {args.modality} brain slice, 7T high field strength, precise anatomical details, sharp focus, medical imaging"
            data_dicts.append(
                {"lr": lr_path, "hr": hr_path, "txt": prompt, "subject_id": subject_id}
            )
    return data_dicts


def get_subject_data_dicts(args: DatasetConfig, verbose=0):
    """Parses BIDS structure to find paired 3T (LR) and 7T (HR) files."""
    lr_dir = args.data_dir / args.lr_base
    hr_dir = args.data_dir / args.hr_base
    data_dicts = []
    if verbose:
        print(f"Scanning for {args.modality} pairs in {data_dir}...")
    # Iterate through 7T subjects (assuming 7T is the target/ground truth)
    for subject_dir in hr_dir.glob("sub-*"):
        subject_id = subject_dir.name
        # Define paths for HR (7T) # Note: BIDS filenames usually look like: sub-01_ses-01_acq-highres_T1w.nii.gz
        # We use glob to find the specific modality file
        hr_files = list((subject_dir / "anat").glob(f"*{args.modality}*.nii.gz"))
        # Check corresponding 3T (LR) subject
        lr_subject_dir = lr_dir / subject_id
        if not lr_subject_dir.exists():
            continue
        lr_files = list((lr_subject_dir / "anat").glob(f"*{args.modality}*.nii.gz"))
        if hr_files and lr_files:
            # Pick the first match found in the anat folder
            hr_path = hr_files[0]
            lr_path = lr_files[0]
            # Refined prompt based on the specific modality
            prompt = f"high quality MRI scan, {args.modality} brain slice, 7T high field strength, precise anatomical details, sharp focus, medical imaging"
            data_dicts.append(
                {
                    "lr": str(lr_path),
                    "hr": str(hr_path),
                    "txt": prompt,
                    "subject_id": subject_id,
                }
            )
    # perform the data split
    generator = torch.Generator().manual_seed(args.seed)
    train, val, test = random_split(
        data_dicts, lengths=args.fractions, generator=generator
    )
    if verbose:
        print(f"Found {len(data_dicts)} paired subjects.")
        print(
            f"Resulted in {len(train)} train subjects, {len(test)} test subjects, {len(val)} validation subjects"
        )
    if args.mode == "train":
        return train
    elif args.mode == "test":
        return test
    elif args.mode == "val":
        return val
    else:
        return data_dicts


def n4_bias_correction(sitk_image, shrink_factor=2):
    """Simple N4 bias correction wrapper (optional)."""
    maskImage = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    inputImage = sitk_image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(inputImage, maskImage)
    return corrected


def rigid_register_and_resample(
    fixed_img_sitk, moving_img_sitk, do_n4=False, verbose=False
):
    """
    Registers `moving` to `fixed` (rigid) and resamples moving into fixed's physical grid.
    Returns: resampled_moving_sitk (SimpleITK image)
    """
    fixed_img_sitk = sitk.Cast(fixed_img_sitk, sitk.sitkFloat32)
    moving_img_sitk = sitk.Cast(moving_img_sitk, sitk.sitkFloat32)
    if do_n4:
        if verbose:
            print("Applying N4 bias correction to fixed and moving.")
        fixed_img_sitk = n4_bias_correction(fixed_img_sitk)
        moving_img_sitk = n4_bias_correction(moving_img_sitk)

    # Initial alignment using geometry center
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img_sitk,
        moving_img_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.Execute(fixed_img_sitk, moving_img_sitk)

    final_transform = registration_method.GetInitialTransform()
    # Actually registration_method modifies the transform; retrieve it:
    try:
        final_transform = registration_method.GetInitialTransform()
    except Exception:
        final_transform = initial_transform

    # Resample moving to fixed grid using computed transform
    resampled_moving = sitk.Resample(
        moving_img_sitk,
        fixed_img_sitk,  # reference image (sets size, spacing, origin, direction)
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img_sitk.GetPixelID(),
    )
    return resampled_moving


def sitk_to_tensor(img_sitk):
    """Convert SITK image (z,y,x) -> torch tensor [1, H, W, D] (MONAI style [C,H,W,D])"""
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # shape (D, H, W)
    # Convert to (H, W, D)
    # We'll keep order (H, W, D) for stacking later; MONAI expects [C, H, W, D]
    D, H, W = arr.shape
    arr = arr.transpose(1, 2, 0)  # (H, W, D)
    # add channel axis -> (1, H, W, D)
    arr = np.expand_dims(arr, axis=0)
    tensor = torch.from_numpy(arr)
    return tensor


def get_valid_z_range_from_mask(mask_sitk, min_fraction=0.01):
    """
    Given a binary mask SITK image [size=(nx,ny,nz)], compute the z-range
    with slices having at least min_fraction of foreground pixels.
    Returns (z0, z1) inclusive indexes in SITK indexing (0..nz-1).
    """
    mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)  # shape (D, H, W)
    D, H, W = mask_np.shape
    per_slice_counts = mask_np.reshape(D, -1).sum(axis=1)  # length D
    min_pixels = int(min_fraction * (H * W))
    # find indices where count >= min_pixels
    good_idx = np.where(per_slice_counts >= min_pixels)[0]
    if len(good_idx) == 0:
        # fallback: use central 50% of slices if Otsu failed
        z0 = D // 4
        z1 = 3 * D // 4
    else:
        z0 = int(good_idx[0])
        z1 = int(good_idx[-1])
    # Convert from SITK array indexing (D,H,W) -> our (H,W,D) indexing when cropping later
    return z0, z1  # indices along D (axial index in sitk array order)


def crop_volume_along_z_np(np_vol_hw_d, z0, z1, axis_is_hw_d=True):
    """
    np_vol_hw_d: numpy array in shape (H, W, D) OR (1,H,W,D) -> keep in mind caller
    We'll return cropped (H, W, D) (or with leading channel preserved).
    The z0,z1 are indices in the D axis (0..D-1).
    """
    TARGET_H = 512
    TARGET_W = 512

    if np_vol_hw_d.ndim == 4:  # (1,H,W,D)
        channel = np_vol_hw_d[0]
        cropped = channel[:, :, z0 : (z1 + 1)]
        return np.expand_dims(cropped, 0)
    elif np_vol_hw_d.ndim == 3:  # (H,W,D)
        return np_vol_hw_d[:, :, z0 : (z1 + 1)]
    else:
        raise ValueError("Unexpected array dims")


def pad_or_center_crop(tensor2d):
    TARGET_H = 512
    TARGET_W = 512
    H, W = tensor2d.shape
    # Center-crop if larger than target
    if H > TARGET_H:
        start_h = (H - TARGET_H) // 2
        tensor2d = tensor2d[start_h : start_h + TARGET_H, :]
        H = TARGET_H
    if W > TARGET_W:
        start_w = (W - TARGET_W) // 2
        tensor2d = tensor2d[:, start_w : start_w + TARGET_W]
        W = TARGET_W

    pad_h = max(0, TARGET_H - H)
    pad_w = max(0, TARGET_W - W)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        import torch.nn.functional as F

        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        tensor2d = F.pad(
            tensor2d.unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        ).squeeze(0)

    return tensor2d


def save_paired_slices(lr_vol, hr_vol, output_dir, prefix):
    """
    Slices 3D volumes along all 3 axes and saves pairs to disk.

    Args:
        lr_vol (np.array): Low Res 3D volume (H, W, D)
        hr_vol (np.array): High Res 3D volume (scaled H, W, D)
        output_dir (str): Root folder to save data
        prefix (str): Identifier for the specific volume (e.g., 'patient_001')
    """

    if torch.is_tensor(lr_vol):
        lr_vol = lr_vol.cpu().numpy()
    if torch.is_tensor(hr_vol):
        hr_vol = hr_vol.cpu().numpy()

    axes_map = {
        #'sagittal': 0, # Slicing from side to side
        #'coronal':  1, # Slicing from front to back
        "axial": 2  # Slicing from top to bottom
    }

    for axis_name, axis_idx in axes_map.items():
        save_path = os.path.join(output_dir, axis_name)
        os.makedirs(save_path, exist_ok=True)

        scale = hr_vol.shape[axis_idx] // lr_vol.shape[axis_idx]

        num_slices = lr_vol.shape[axis_idx]

        print(
            f"Processing {prefix} | {axis_name} | Scale {scale} | num slices {num_slices} | LR Shape: {lr_vol.shape} | HR Shape: {hr_vol.shape}"
        )

        for i in range(num_slices):

            idx_lr = [slice(None)] * 3
            idx_lr[axis_idx] = i

            idx_hr = [slice(None)] * 3
            idx_hr[axis_idx] = i * scale

            # Extract slices
            # tuple(idx_lr) converts the list of slices to a format numpy accepts
            slice_lr = lr_vol[tuple(idx_lr)]
            slice_hr = hr_vol[tuple(idx_hr)]

            filename = f"{axis_name}_{prefix}_{i:04d}.npz"
            full_path = os.path.join(save_path, filename)

            np.savez_compressed(full_path, lr=slice_lr, hr=slice_hr)


if __name__ == "__main__":

    data_dir = Path(
        "data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/Data"
    )
    output_dir = Path(
        "data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/processed"
    )

    try:
        train_ds = PairedMRIDataset(config=DatasetConfig(data_dir=data_dir), verbose=1)
        print(len(train_ds), "paired scans found in the dataset.")

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

        for i, batch in enumerate(train_loader):
            lr_img_3d = batch["lr"][0, 0]
            hr_img_3d = batch["hr"][0, 0]

            # Generate a unique name
            vol_name = f"vol_{i:03d}"

            # Run generation
            save_paired_slices(lr_img_3d, hr_img_3d, output_dir, vol_name)

        print(f"\nProcessing complete. Data saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
