import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

def get_data_dicts_artificial(data_dir, modality="T2w"):
    """
    Parses BIDS structure to find 3T instances.
    Sets 'hr' as the original 3T path and 'lr' as the same 3T path.
    The actual downsampling/degradation should be performed in the
    Dataset's __getitem__ or preprocessing step.
    """
    data_dir = Path(data_dir)
    # We only need the 3T base for this specific task
    lr_base = data_dir / 'rawdata_BIDS_3T'
    data_dicts = []

    # Iterate through 3T subjects
    for subject_dir in lr_base.glob('sub-*'):
        subject_id = subject_dir.name

        # Find the 3T files for the specific modality
        t3_files = list((subject_dir / 'anat').glob(f'*{modality}*.nii*'))

        if t3_files:
            # Both HR and LR point to the 3T file
            # The Dataset class will take this path and create a degraded version for 'lr'
            t3_path = str(t3_files[0])

            # Updated prompt: removed "7T" since we are using 3T as ground truth
            prompt = f"high quality MRI scan, {modality} brain slice, 3T field strength, precise anatomical details, sharp focus, medical imaging"

            data_dicts.append({
                'lr': t3_path,         # This will be downsampled later
                'hr': t3_path,         # This remains the original target
                'txt': prompt,
                'subject_id': subject_id
            })

    print(f"Found {len(data_dicts)} 3T subjects for downsampling task.")
    return data_dicts

def n4_bias_correction(sitk_image, shrink_factor=2):
    """Simple N4 bias correction wrapper (optional)."""
    maskImage = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    inputImage = sitk_image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(inputImage, maskImage)
    return corrected


def rigid_register_and_resample(fixed_img_sitk, moving_img_sitk, do_n4=False, verbose=False):
    """
    Registers `moving` to `fixed` (rigid) and resamples moving into fixed's physical grid.
    Returns: resampled_moving_sitk (SimpleITK image)
    """
    fixed_img_sitk  = sitk.Cast(fixed_img_sitk,  sitk.sitkFloat32)
    moving_img_sitk = sitk.Cast(moving_img_sitk, sitk.sitkFloat32)
    if do_n4:
        if verbose: print("Applying N4 bias correction to fixed and moving.")
        fixed_img_sitk = n4_bias_correction(fixed_img_sitk)
        moving_img_sitk = n4_bias_correction(moving_img_sitk)

    # Initial alignment using geometry center
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img_sitk,
        moving_img_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
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
        gradientMagnitudeTolerance=1e-8
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
        fixed_img_sitk,            # reference image (sets size, spacing, origin, direction)
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img_sitk.GetPixelID()
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
        cropped = channel[:, :, z0:(z1+1)]
        return np.expand_dims(cropped, 0)
    elif np_vol_hw_d.ndim == 3:  # (H,W,D)
        return np_vol_hw_d[:, :, z0:(z1+1)]
    else:
        raise ValueError("Unexpected array dims")

def pad_or_center_crop(tensor2d):
        TARGET_H = 512
        TARGET_W = 512
        H, W = tensor2d.shape
        # Center-crop if larger than target
        if H > TARGET_H:
            start_h = (H - TARGET_H) // 2
            tensor2d = tensor2d[start_h:start_h + TARGET_H, :]
            H = TARGET_H
        if W > TARGET_W:
            start_w = (W - TARGET_W) // 2
            tensor2d = tensor2d[:, start_w:start_w + TARGET_W]
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
            tensor2d = F.pad(tensor2d.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0).squeeze(0)

        return tensor2d


class SliceDataset(Dataset):
    def __init__(self, pairs, slice_axis=2, cache_dir='./cache', do_registration=True, do_n4=False, lr_clip=(0, 2000), hr_clip=(0, 900)):
        """
        Args:
            pairs: A list of dicts (from get_data_dicts_artificial) to be used by this instance.
            slice_axis: 0=sagittal,1=coronal,2=axial
            cache_dir: where to store resampled numpy .npz caches per subject
            do_registration: perform rigid registration LR->HR (recommended True)
            do_n4: apply N4 bias correction prior to registration (optional)
            intensity_clip: (a_min, a_max) for intensity normalization to [0,1]
        """
        self.pairs = pairs
        self.slice_axis = slice_axis
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.do_registration = do_registration
        self.do_n4 = do_n4
        self.lr_clip = lr_clip
        self.hr_clip = hr_clip

        if not self.pairs:
            raise ValueError("No pairs found. Check paths.")

        # Preprocess & cache
        self.slice_metadata = []
        self._prepare_all_pairs()

    def _prepare_all_pairs(self):
        print("Preparing pairs (registration/resampling + caching).")
        for item in self.pairs:
            sid = item['subject_id']
            if sid == 'sub-15':
              print(f'Skipping subject 15 due to wrong layout')
              continue
            cache_file = self.cache_dir / f"{sid}_resampled.npz"
            if cache_file.exists():
                npz = np.load(cache_file)
                hr_arr = npz['hr']  # shape [1,H,W,D]
                lr_arr = npz['lr']
            else:
                # Read with SimpleITK
                hr_sitk = sitk.ReadImage(item['hr'])
                lr_sitk = sitk.ReadImage(item['lr'])

                # Optionally register and resample
                if self.do_registration:
                    try:
                        lr_resampled_sitk = rigid_register_and_resample(hr_sitk, lr_sitk, do_n4=self.do_n4, verbose=False)
                    except Exception as e:
                        print(f"Registration failed for {sid}: {e}. Falling back to simple resample onto HR grid.")
                        lr_resampled_sitk = sitk.Resample(lr_sitk, hr_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, lr_sitk.GetPixelID())
                else:
                    # Direct resample (no registration) onto HR grid
                    lr_resampled_sitk = sitk.Resample(lr_sitk, hr_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, lr_sitk.GetPixelID())

                # Optionally resample HR itself to a smaller target spacing if wanted.
                # For now we keep HR as original 7T grid (so hr_sitk defines canonical grid)

                # Convert to numpy/tensors (and normalize intensities)
                hr_tensor = sitk_to_tensor(hr_sitk)   # [1,H,W,D]
                lr_tensor = sitk_to_tensor(lr_resampled_sitk)

                hr_arr = hr_tensor.numpy().astype(np.float32)
                lr_arr = lr_tensor.numpy().astype(np.float32)

                # Crop first/last 30 slices along chosen slice_axis to remove 'air'
                dim_idx = self.slice_axis + 1  # mapping into [C,H,W,D] array
                num_slices = hr_arr.shape[dim_idx]
                crop_start = 80
                crop_end = num_slices - 30
                if crop_end <= crop_start or num_slices <= 60:
                    # Fallback: don't crop if not enough slices
                    print(f"Subject {sid}: volume too small ({num_slices} slices) to crop 30/30, skipping crop.")
                else:
                    slicer = [slice(None)] * hr_arr.ndim
                    slicer[dim_idx] = slice(crop_start, crop_end)
                    hr_arr = hr_arr[tuple(slicer)]
                    lr_arr = lr_arr[tuple(slicer)]

                # Intensity clipping and scaling to [0,1]
                # a_min, a_max = self.intensity_clip
                # hr_arr = hr_tensor.numpy().astype(np.float32)
                # lr_arr = lr_tensor.numpy().astype(np.float32)
                # hr_arr = np.clip((hr_arr - a_min) / (a_max - a_min), 0.0, 1.0)
                # lr_arr = np.clip((lr_arr - a_min) / (a_max - a_min), 0.0, 1.0)

                a_min_hr, a_max_hr = float(self.hr_clip[0]), float(self.hr_clip[1])
                a_min_lr, a_max_lr = float(self.lr_clip[0]), float(self.lr_clip[1])
                # final clamp and cast
                hr_arr = np.clip((hr_arr - a_min_hr) / (a_max_hr - a_min_hr), 0.0, 1.0).astype(np.float32)
                lr_arr = np.clip((lr_arr - a_min_lr) / (a_max_lr - a_min_lr), 0.0, 1.0).astype(np.float32)

                # Save cache
                np.savez_compressed(cache_file, hr=hr_arr, lr=lr_arr)

            # Now create slice metadata entries
            # hr_arr shape: [1, H, W, D]
            num_slices = hr_arr.shape[self.slice_axis + 1]  # channel + dims
            for s_idx in range(num_slices):
                self.slice_metadata.append({
                    'hr_arr': hr_arr,
                    'lr_arr': lr_arr,
                    'slice_idx': int(s_idx),
                    'txt': item['txt'],
                    'subject_id': sid
                })

    def _prepare_all_pairs(self):
        print("Preparing pairs (3T Downsampling + caching).")
        # Define the downsampling factor (e.g., 4x)
        DS_FACTOR = 4.0

        for item in self.pairs:
            sid = item['subject_id']
            if sid == 'sub-15': continue

            cache_file = self.cache_dir / f"{sid}_3T_downsampled.npz"
            if cache_file.exists():
                npz = np.load(cache_file)
                hr_arr = npz['hr']
                lr_arr = npz['lr']
            elif item['lr'] == item['hr']:
                # 1. Load the 3T image (item['hr'] and item['lr'] both point here now)
                hr_sitk = sitk.ReadImage(item['hr'])
                hr_sitk = sitk.Cast(hr_sitk, sitk.sitkFloat32)

                # 2. Create the Low-Res (LR) version
                # Anti-aliasing: Blur before downsampling to simulate lower-res physics
                sigma = (DS_FACTOR / 2.0)
                blurred_sitk = sitk.DiscreteGaussian(hr_sitk, variance=sigma**2)

                # Calculate new dimensions for the "true" low-res space
                orig_spacing = hr_sitk.GetSpacing()
                orig_size = hr_sitk.GetSize()
                new_spacing = [s * DS_FACTOR for s in orig_spacing]
                new_size = [int(sz / DS_FACTOR) for sz in orig_size]

                # Step A: Resample down to low-res grid
                lr_low_res = sitk.Resample(
                    blurred_sitk,
                    new_size,
                    sitk.Transform(),
                    sitk.sitkLinear,
                    hr_sitk.GetOrigin(),
                    new_spacing,
                    hr_sitk.GetDirection(),
                    0.0,
                    hr_sitk.GetPixelID()
                )

                # Step B: Resample back up to the HR grid (so tensors match shape)
                # We use Linear interpolation to simulate what a basic upscaler would do
                lr_resampled_sitk = sitk.Resample(
                    lr_low_res,
                    hr_sitk, # Use original HR as reference grid
                    sitk.Transform(),
                    sitk.sitkLinear,
                    0.0,
                    hr_sitk.GetPixelID()
                )

                # 3. Convert to numpy/tensors
                hr_tensor = sitk_to_tensor(hr_sitk)
                lr_tensor = sitk_to_tensor(lr_resampled_sitk)

                hr_arr = hr_tensor.numpy().astype(np.float32)
                lr_arr = lr_tensor.numpy().astype(np.float32)

                # 4. Crop slices along chosen slice_axis (same as your original code)
                dim_idx = self.slice_axis + 1
                num_slices = hr_arr.shape[dim_idx]
                crop_start, crop_end = 80, num_slices - 30

                if crop_end > crop_start and num_slices > 60:
                    slicer = [slice(None)] * hr_arr.ndim
                    slicer[dim_idx] = slice(crop_start, crop_end)
                    hr_arr = hr_arr[tuple(slicer)]
                    lr_arr = lr_arr[tuple(slicer)]

                # 5. Intensity clipping and scaling to [0,1]
                a_min_hr, a_max_hr = float(self.hr_clip[0]), float(self.hr_clip[1])
                a_min_lr, a_max_lr = float(self.lr_clip[0]), float(self.lr_clip[1])

                hr_arr = np.clip((hr_arr - a_min_hr) / (a_max_hr - a_min_hr), 0.0, 1.0).astype(np.float32)
                lr_arr = np.clip((lr_arr - a_min_lr) / (a_max_lr - a_min_lr), 0.0, 1.0).astype(np.float32)

                # Save cache
                np.savez_compressed(cache_file, hr=hr_arr, lr=lr_arr)
            else:
                # Read with SimpleITK
                hr_sitk = sitk.ReadImage(item['hr'])
                lr_sitk = sitk.ReadImage(item['lr'])

                # Optionally register and resample
                if self.do_registration:
                    try:
                        lr_resampled_sitk = rigid_register_and_resample(hr_sitk, lr_sitk, do_n4=self.do_n4, verbose=False)
                    except Exception as e:
                        print(f"Registration failed for {sid}: {e}. Falling back to simple resample onto HR grid.")
                        lr_resampled_sitk = sitk.Resample(lr_sitk, hr_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, lr_sitk.GetPixelID())
                else:
                    # Direct resample (no registration) onto HR grid
                    lr_resampled_sitk = sitk.Resample(lr_sitk, hr_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, lr_sitk.GetPixelID())

                # Optionally resample HR itself to a smaller target spacing if wanted.
                # For now we keep HR as original 7T grid (so hr_sitk defines canonical grid)

                # Convert to numpy/tensors (and normalize intensities)
                hr_tensor = sitk_to_tensor(hr_sitk)   # [1,H,W,D]
                lr_tensor = sitk_to_tensor(lr_resampled_sitk)

                hr_arr = hr_tensor.numpy().astype(np.float32)
                lr_arr = lr_tensor.numpy().astype(np.float32)

                # Crop first/last 30 slices along chosen slice_axis to remove 'air'
                dim_idx = self.slice_axis + 1  # mapping into [C,H,W,D] array
                num_slices = hr_arr.shape[dim_idx]
                crop_start = 80
                crop_end = num_slices - 30
                if crop_end <= crop_start or num_slices <= 60:
                    # Fallback: don't crop if not enough slices
                    print(f"Subject {sid}: volume too small ({num_slices} slices) to crop 30/30, skipping crop.")
                else:
                    slicer = [slice(None)] * hr_arr.ndim
                    slicer[dim_idx] = slice(crop_start, crop_end)
                    hr_arr = hr_arr[tuple(slicer)]
                    lr_arr = lr_arr[tuple(slicer)]

                # Intensity clipping and scaling to [0,1]
                # a_min, a_max = self.intensity_clip
                # hr_arr = hr_tensor.numpy().astype(np.float32)
                # lr_arr = lr_tensor.numpy().astype(np.float32)
                # hr_arr = np.clip((hr_arr - a_min) / (a_max - a_min), 0.0, 1.0)
                # lr_arr = np.clip((lr_arr - a_min) / (a_max - a_min), 0.0, 1.0)

                a_min_hr, a_max_hr = float(self.hr_clip[0]), float(self.hr_clip[1])
                a_min_lr, a_max_lr = float(self.lr_clip[0]), float(self.lr_clip[1])
                # final clamp and cast
                hr_arr = np.clip((hr_arr - a_min_hr) / (a_max_hr - a_min_hr), 0.0, 1.0).astype(np.float32)
                lr_arr = np.clip((lr_arr - a_min_lr) / (a_max_lr - a_min_lr), 0.0, 1.0).astype(np.float32)

                # Save cache
                np.savez_compressed(cache_file, hr=hr_arr, lr=lr_arr)


            # Create slice metadata entries...
            num_slices = hr_arr.shape[self.slice_axis + 1]
            for s_idx in range(num_slices):
                self.slice_metadata.append({
                    'hr_arr': hr_arr, 'lr_arr': lr_arr, 'slice_idx': int(s_idx),
                    'txt': item['txt'], 'subject_id': sid
                })

    def __len__(self):
        return len(self.slice_metadata)

    def __getitem__(self, idx):
        m = self.slice_metadata[idx]
        hr = torch.from_numpy(m['hr_arr']).float()  # [1,H,W,D]
        lr = torch.from_numpy(m['lr_arr']).float()
        s = m['slice_idx']

        # Build slicing tuple for [C, H, W, D]
        slicer = [slice(None), slice(None), slice(None), slice(None)]
        slicer[self.slice_axis + 1] = s
        hr_slice = hr[tuple(slicer)].squeeze(0)  # [H, W] or [H, W] if 2D slice
        lr_slice = lr[tuple(slicer)].squeeze(0)
        #hr_flat = hr_slice.reshape(-1)
        ## guard against empty / degenerate slices
        #if hr_flat.numel() > 0:
        #    p_low = torch.quantile(hr_flat, 0.01)
        #    p_high = torch.quantile(hr_flat, 0.99)
        #    if p_high > p_low:
        #        hr_slice = (hr_slice - p_low) / (p_high - p_low)
        #    else:
        #        # constant slice -> zeros
        #        hr_slice = torch.zeros_like(hr_slice)
        #hr_slice = torch.clamp(hr_slice, 0.0, 1.0)

        hr_slice = pad_or_center_crop(hr_slice)
        lr_slice = pad_or_center_crop(lr_slice)

        # If you want explicit shape [1,H,W], add channel back
        hr_slice = hr_slice.unsqueeze(0)  # [1,H,W]
        lr_slice = lr_slice.unsqueeze(0)

        return {'hr': hr_slice, 'lr': lr_slice, 'txt': m['txt'], 'subject_id': m['subject_id']}