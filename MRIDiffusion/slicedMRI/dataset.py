import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import SimpleITK as sitk
from typing import Dict, Union, Tuple
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image
import pydicom
import json


from MRIDiffusion.slicedMRI.transform_to_2D_slices import (
    get_subject_data_dicts,
    n4_bias_correction,
    rigid_register_and_resample,
    sitk_to_tensor,
    get_valid_z_range_from_mask,
    crop_volume_along_z_np,
    pad_or_center_crop,
)
from MRIDiffusion.slicedMRI.config import DatasetConfig


class FastMRILazyDataset(Dataset):
    """
    Lazy-loading dataset for FastMRI brain DICOMs.
    Simulates Low-Field (1T) from High-Field (3T) via downsampling.
    """

    def __init__(
        self,
        config: DatasetConfig,
    ):
        self.target_size = config.target_size
        self.scale_factor = config.scale_factor
        with open(config.fastMRI_manifest_json, "r") as f:
            self.all_patient_records = json.load(f)
        self.subjects = self._get_filtered_subjects(
            config.contrast_filter,
            config.strength_filter,
            config.seed,
            config.fractions,
            config.mode,
        )
        self.slice_metadata = []
        self._prepare_slice_index()

    def _get_filtered_subjects(self, contrast, strength, seed, fractions, mode):
        """Filters subjects by physics and performs patient-level split."""
        valid_subjects = []
        for pid, strengths in self.all_patient_records.items():
            if strength in strengths and contrast in strengths[strength]:
                valid_subjects.append(
                    {
                        "subject_id": pid,
                        "strength": strength,
                        "contrast": contrast,
                        "txt": f"high quality {contrast} brain MRI, {strength} field strength, medical imaging",
                    }
                )
        # Subject-level split
        generator = torch.Generator().manual_seed(seed)
        train, val, test = random_split(
            valid_subjects, lengths=fractions, generator=generator
        )
        mapping = {"train": train, "val": val, "test": test}
        selected = mapping.get(mode, train)
        return [selected.dataset[i] for i in selected.indices]

    def _prepare_slice_index(self):
        """Creates a flat list of pointers to every slice for lazy access."""
        for item in self.subjects:
            pid = item["subject_id"]
            strength = item["strength"]
            contrast = item["contrast"]
            slices = self.all_patient_records[pid][strength][contrast]
            for s_info in slices:
                self.slice_metadata.append(
                    {
                        "path": s_info["filename"],
                        "subject_id": pid,
                        "txt": item["txt"],
                        "instance": s_info["instanceNumber"],
                    }
                )

    def _pad_to_target(self, arr: np.ndarray) -> np.ndarray:
        """Center pads the image to target_size without resizing anatomy."""
        h, w = arr.shape
        th, tw = self.target_size
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        # If image is larger than target, we crop the center
        if h > th or w > tw:
            start_h = max(0, (h - th) // 2)
            start_w = max(0, (w - tw) // 2)
            arr = arr[start_h : start_h + th, start_w : start_w + tw]
            h, w = arr.shape
            pad_h = th - h
            pad_w = tw - w
        padding = (
            (pad_h // 2, pad_h - (pad_h // 2)),
            (pad_w // 2, pad_w - (pad_w // 2)),
        )
        return np.pad(arr, padding, mode="constant", constant_values=0)

    def _simulate_low_res(self, hr_arr: np.ndarray) -> np.ndarray:
        """Simulates low field/resolution via Gaussian blur and down-up sampling."""
        # Blur to simulate lower SNR and point spread function
        sigma = 0.5 * self.scale_factor
        blurred = gaussian_filter(hr_arr, sigma=sigma)
        # Downsample then upsample back to target size (Bicubic)
        pil_img = Image.fromarray(blurred)
        small_size = (
            int(self.target_size[1] // self.scale_factor),
            int(self.target_size[0] // self.scale_factor),
        )
        lr_img = pil_img.resize(small_size, resample=Image.BICUBIC)
        lr_up = lr_img.resize(self.target_size, resample=Image.BICUBIC)
        return np.array(lr_up)

    def __len__(self):
        return len(self.slice_metadata)

    def __getitem__(self, idx) -> Dict:
        meta = self.slice_metadata[idx]
        # Lazy load DICOM
        ds = pydicom.dcmread(meta["path"])
        arr = ds.pixel_array.astype(np.float32)
        # Normalize 0-1
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        hr_arr = self._pad_to_target(arr)
        lr_arr = self._simulate_low_res(hr_arr)
        return {
            "hr": torch.from_numpy(hr_arr).unsqueeze(0).float(),
            "lr": torch.from_numpy(lr_arr).unsqueeze(0).float(),
            "txt": meta["txt"],
            "subject_id": meta["subject_id"],
            "instance": meta["instance"],
        }


class PairedMRI_MiniDataset(Dataset):
    """
    Dataset for 64mT to 3T paired MRI scans for 11 subjects with 24 slices each (total of 264 slices).
    Utilizes `FLAIR` modality as the subject scans are aligned on the axial slices
    """

    def __init__(
        self,
        config: DatasetConfig,
        verbose: int = 0,
    ):
        """
        - `config: DatasetConfig`
        - `verbose: int = 0`
        """
        self.root_dir = config.data_dir
        self.verbose = verbose
        self.subjects = self._get_subject_pairs(
            seed=config.seed,
            fractions=config.fractions,
            mode=config.mode,
        )
        self.cache_dir = Path(config.cacha_dir_str)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        self.slice_metadata = []
        self._prepare_pairs(config)

    def _get_subject_pairs(
        self, seed: int, fractions: tuple[float, float, float], mode: str
    ) -> list[dict[str, Union[Path, str]]]:
        dicts = []
        for subject_path in self.root_dir.glob("sub-*"):
            subject_id = subject_path.name
            hr_file = subject_path / "anat" / f"{subject_id}_acq-highres_FLAIR.nii.gz"
            lr_file = subject_path / "anat" / f"{subject_id}_acq-lowres_FLAIR.nii.gz"
            if lr_file.exists() and hr_file.exists():
                dicts.append(
                    {
                        "lr": lr_file,
                        "hr": hr_file,
                        "txt": f"high quality MRI scan, FLAIR brain slice, 3T high field strength, precise anatomical details, sharp focus, medical imaging",
                        "subject_id": subject_id,
                    }
                )
        # perform subject level split
        generator = torch.Generator().manual_seed(seed)
        train, val, test = random_split(dicts, lengths=fractions, generator=generator)
        if mode == "train":
            return [train.dataset[i] for i in train.indices]
        elif mode == "test":
            return [test.dataset[i] for i in test.indices]
        elif mode == "val":
            return [val.dataset[i] for i in val.indices]
        else:
            return dicts

    def _prepare_pairs(self, config: DatasetConfig) -> None:
        num_slices_per_subject = 24
        for item in self.subjects:
            sid = item["subject_id"]
            cache_file = self.cache_dir / f"minids_{sid}.npz"
            if cache_file.exists():
                npz = np.load(cache_file)
                hr_numpy = npz["hr"]  # shape [1,D,H,W]
                lr_numpy = npz["lr"]
            else:
                # Read with SimpleITK
                hr_sitk = sitk.ReadImage(item["hr"])
                hr_numpy = sitk.GetArrayFromImage(hr_sitk)
                hr_numpy = np.clip(
                    (hr_numpy - hr_numpy.min()) / (hr_numpy.max() - hr_numpy.min()),
                    0.0,
                    1.0,
                )[
                    :num_slices_per_subject, :, :
                ]  # images only offer information on first 24 slices
                lr_sitk = sitk.ReadImage(item["lr"])
                lr_numpy = scipy.ndimage.zoom(
                    sitk.GetArrayFromImage(lr_sitk), (1, 3.2, 3.2), order=0
                )
                lr_numpy = np.clip(
                    (lr_numpy - lr_numpy.min()) / (lr_numpy.max() - lr_numpy.min()),
                    0.0,
                    1.0,
                )[
                    :num_slices_per_subject, :, :
                ]  # images only offer information on first 24 slices
                np.savez_compressed(cache_file, hr=hr_numpy, lr=lr_numpy)
            for s_idx in range(num_slices_per_subject):
                self.slice_metadata.append(
                    {
                        "hr_arr": hr_numpy[s_idx, :, :],
                        "lr_arr": lr_numpy[s_idx, :, :],
                        "slice_idx": int(s_idx),
                        "txt": item["txt"],
                        "subject_id": sid,
                    }
                )

    def __len__(self):
        return len(self.slice_metadata)

    def __getitem__(self, idx) -> Dict:
        meta = self.slice_metadata[idx]
        hr = torch.from_numpy(meta["hr_arr"]).float()  # [1,H,W]
        lr = torch.from_numpy(meta["lr_arr"]).float()  # [1,H,W]
        return {
            "hr": hr,
            "lr": lr,
            "txt": meta["txt"],
            "subject_id": meta["subject_id"],
        }


class PairedMRIDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        verbose: int = 0,
    ):
        """
        - `config: DatasetConfig`
        - `verbose: int = 0`
        """
        self.root_dir = config.data_dir
        self.slice_axis = config.slice_axis
        self.cache_dir = Path(config.cacha_dir_str)
        if not Path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.do_registration = config.do_registration
        self.do_n4 = config.do_n4
        self.lr_clip = config.lr_intensity_clip
        self.hr_clip = config.hr_intensity_clip
        self.verbose = verbose
        self.pairs = get_subject_data_dicts(args=config, verbose=verbose)
        if not self.pairs:
            raise ValueError("No pairs found. Check paths.")

        # Preprocess & cache
        self.slice_metadata = []
        self._prepare_all_pairs()

    def _prepare_all_pairs(self):
        if self.verbose:
            print("Preparing pairs (registration/resampling + caching).")
        for item in self.pairs:
            sid = item["subject_id"]
            if sid == "sub-15":
                if self.verbose:
                    print(f"Skipping subject 15 due to wrong layout")
                continue
            if sid != "sub-08":
                if self.verbose:
                    print(f"skipping {sid}")
                continue
            cache_file = self.cache_dir / f"{sid}_resampled.npz"
            if cache_file.exists():
                npz = np.load(cache_file)
                hr_arr = npz["hr"]  # shape [1,H,W,D]
                lr_arr = npz["lr"]
            else:
                # Read with SimpleITK
                hr_sitk = sitk.ReadImage(item["hr"])
                lr_sitk = sitk.ReadImage(item["lr"])

                # Optionally register and resample
                if self.do_registration:
                    try:
                        lr_resampled_sitk = rigid_register_and_resample(
                            hr_sitk, lr_sitk, do_n4=self.do_n4, verbose=False
                        )
                    except Exception as e:
                        print(
                            f"Registration failed for {sid}: {e}. Falling back to simple resample onto HR grid."
                        )
                        lr_resampled_sitk = sitk.Resample(
                            lr_sitk,
                            hr_sitk,
                            sitk.Transform(),
                            sitk.sitkLinear,
                            0.0,
                            lr_sitk.GetPixelID(),
                        )
                else:
                    # Direct resample (no registration) onto HR grid
                    lr_resampled_sitk = sitk.Resample(
                        lr_sitk,
                        hr_sitk,
                        sitk.Transform(),
                        sitk.sitkLinear,
                        0.0,
                        lr_sitk.GetPixelID(),
                    )

                # Optionally resample HR itself to a smaller target spacing if wanted.
                # For now we keep HR as original 7T grid (so hr_sitk defines canonical grid)

                # Convert to numpy/tensors (and normalize intensities)
                hr_tensor = sitk_to_tensor(hr_sitk)  # [1,H,W,D]
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
                    print(
                        f"Subject {sid}: volume too small ({num_slices} slices) to crop 30/30, skipping crop."
                    )
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
                hr_arr = np.clip(
                    (hr_arr - a_min_hr) / (a_max_hr - a_min_hr), 0.0, 1.0
                ).astype(np.float32)
                lr_arr = np.clip(
                    (lr_arr - a_min_lr) / (a_max_lr - a_min_lr), 0.0, 1.0
                ).astype(np.float32)

                # Save cache
                np.savez_compressed(cache_file, hr=hr_arr, lr=lr_arr)

            # Now create slice metadata entries
            # hr_arr shape: [1, H, W, D]
            num_slices = hr_arr.shape[self.slice_axis + 1]  # channel + dims
            for s_idx in range(num_slices):
                self.slice_metadata.append(
                    {
                        "hr_arr": hr_arr,
                        "lr_arr": lr_arr,
                        "slice_idx": int(s_idx),
                        "txt": item["txt"],
                        "subject_id": sid,
                    }
                )

    def __len__(self):
        return len(self.slice_metadata)

    def __getitem__(self, idx) -> Dict:
        m = self.slice_metadata[idx]
        hr = torch.from_numpy(m["hr_arr"]).float()  # [1,H,W,D]
        lr = torch.from_numpy(m["lr_arr"]).float()
        s = m["slice_idx"]

        # Build slicing tuple for [C, H, W, D]
        slicer = [slice(None), slice(None), slice(None), slice(None)]
        slicer[self.slice_axis + 1] = s
        hr_slice = hr[tuple(slicer)].squeeze(0)  # [H, W] or [H, W] if 2D slice
        lr_slice = lr[tuple(slicer)].squeeze(0)
        # hr_flat = hr_slice.reshape(-1)
        ## guard against empty / degenerate slices
        # if hr_flat.numel() > 0:
        #    p_low = torch.quantile(hr_flat, 0.01)
        #    p_high = torch.quantile(hr_flat, 0.99)
        #    if p_high > p_low:
        #        hr_slice = (hr_slice - p_low) / (p_high - p_low)
        #    else:
        #        # constant slice -> zeros
        #        hr_slice = torch.zeros_like(hr_slice)
        # hr_slice = torch.clamp(hr_slice, 0.0, 1.0)

        hr_slice = pad_or_center_crop(hr_slice)
        lr_slice = pad_or_center_crop(lr_slice)

        # If you want explicit shape [1,H,W], add channel back
        hr_slice = hr_slice.unsqueeze(0)  # [1,H,W]
        lr_slice = lr_slice.unsqueeze(0)

        return {
            "hr": hr_slice,
            "lr": lr_slice,
            "txt": m["txt"],
            "subject_id": m["subject_id"],
        }


def show_batch(dataloader):
    """
    Visualizes a single batch from the loader.
    """
    batch = next(iter(dataloader))
    lr_imgs = batch["lr"]
    hr_imgs = batch["hr"]

    # Grab the first item in the batch
    lr = lr_imgs[0].squeeze()  # remove channel dim for plotting
    hr = hr_imgs[0].squeeze()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Low Res (Input)\nShape: {lr.shape}")
    plt.imshow(lr, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"High Res (Target)\nShape: {hr.shape}")
    plt.imshow(hr, cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    PROCESSED_DIR = Path(
        "data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/processed"
    )

    train_ds = PairedMRIDataset(
        config=DatasetConfig(data_dir=PROCESSED_DIR, slice_axis=2), verbose=1
    )

    print(train_ds[0]["lr"].shape, train_ds[0]["hr"].shape)
    print(train_ds[1]["lr"].shape, train_ds[1]["hr"].shape)
    print(train_ds[2]["lr"].shape, train_ds[2]["hr"].shape)

    if len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

        print(f"DataLoader ready with batch size 8.")

        try:
            show_batch(train_loader)

            print("\nSimulating one iteration...")
            batch = next(iter(train_loader))
            lr = batch["lr"]
            hr = batch["hr"]
            print(f"Batch LR Shape: {lr.shape}")  # Should be [8, 1, H, W]
            print(f"Batch HR Shape: {hr.shape}")  # Should be [8, 1, H_scale, W_scale]

        except Exception as e:
            print(f"Error during loading: {e}")
    else:
        print("No files found. Please check the PROCESSED_DIR path.")
