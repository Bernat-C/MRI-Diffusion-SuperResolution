from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    - `root_dir`: dataset root (contains rawdata_BIDS_3T and rawdata_BIDS_7T)
    - `lr_base: str = "rawdata_BIDS_3T": The sub-folder for the resolution
    - `hr_base: str = "rawdata_BIDS_7T": The sub-folder for the resolution
    - `slice_axis=2`: 0=sagittal,1=coronal,2=axial
    - `cache_dir="./cache/"`: where to store resampled numpy .npz caches per subject
    - `mode="train"`: can be `train`, `val`, or `test`, performs a subject level split
    - `fractions=[0.8, 0.0, 0.2]`: fractions of `subjects` for each split
    - `seed=42`: seed to use for splitting
    - `target_size = (512,512)`: final pixel size to crop or pad the MRI scan to
    - `contrast_filter = "T2"`: filter for the MRI contrast to be used on the fastMRI dataset (T2 has the most entries by a factor of 100x)
    - `strength_filter = "3.0T"`: filter for the MRI magnet field strenght to be used on the fastMRI dataset (3T or 1.5T)
    - `scale_factor = 4.0`: factor by which to scale donw HF scans to obtain faked LF scans
    - `fastMRI_manifest_json = ""`: Path to the manifest json ofbthe fastMRI dataset
    - ` modality= "T2w"`: The modality (=MRI pulse sequence to highlight differences in the T2 relaxation time of tissues) to use. Can be `T2w` or `T1w`
    - `do_registration=bool`: perform rigid registration LR->HR
    - `do_n4=True`: apply N4 bias correction prior to registration (optional)
    - `lr_intensity_clip=(0, 2000)`: (a_min, a_max) for intensity normalization to [0,1]
    - `hr_intensity_clip=(0, 900)`: (a_min, a_max) for intensity normalization to [0,1]
    """

    data_dir: Path = Path("./mri_dataset/")
    lr_base: str = "rawdata_BIDS_3T"
    hr_base: str = "rawdata_BIDS_7T"
    cacha_dir_str: str = "./cache"
    mode: str = "train"  # can be train val or test
    fractions: tuple[float, float, float] = (
        0.8,
        0.0,
        0.2,
    )  # fractions of subjects for each of the three splits
    seed: int = 42
    target_size: tuple[int, int] = (512, 512)
    contrast_filter: str = "T2"
    strength_filter: str = "3.0T"
    scale_factor: float = 4.0
    fastMRI_manifest_json: str = ""
    modality: str = "T2w"
    slice_axis: int = 2
    do_registration: bool = True
    do_n4: bool = True
    lr_intensity_clip: tuple[int] = (0, 2000)
    hr_intensity_clip: tuple[int] = (0, 900)
