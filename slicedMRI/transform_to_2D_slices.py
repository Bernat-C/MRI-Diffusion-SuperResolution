import os
import numpy as np
import torch
from tqdm import tqdm # Install with: pip install tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstd,
    OrientationD,
    ScaleIntensityRangeD,
    ResizeD,
    ToTensorD
)

def get_data_dicts(data_dir):
  lr_dir = data_dir / '64mT data'
  hr_dir = data_dir / '3T data'

  data_dicts = []

  print("Scanning for data...")
  for subject_dir in lr_dir.glob('sub-*'):
      subject_id = os.path.basename(subject_dir)
      #print(f" ---- SUBJECT {subject_id} ---- ")
      sess_dirs = list(subject_dir.glob('ses-*'))
      session_dir = False if not sess_dirs else sess_dirs[0]
      if session_dir:
          anat_dir = session_dir / 'anat'

          # Find the NIfTI file in the LR anat folder
          # Use .search() if you know the suffix, or .glob()
          lr_files = list(anat_dir.glob('*T1w.nii.gz'))
          if not lr_files:
              continue # Skip if no NIfTI file

          lr_nifti_path = lr_files[0]
          hr_name = f"{subject_id}_acq-highres_T1w.nii.gz"
          hr_nifti_path = hr_dir / subject_dir.name / 'anat' / hr_name

          # Only add the pair if the high-res file also exists
          if hr_nifti_path.exists():
              subject = {
                  'lr': str(lr_nifti_path),
                  'hr': str(hr_nifti_path)
              }
              data_dicts.append(subject)
  return data_dicts

class PairedMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.file_pairs = get_data_dicts(root_dir)

        if not self.file_pairs:
            print("Warning: get_data_dicts() returned no file pairs.")
        else:
            print(f"Found {len(self.file_pairs)} paired scans.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_item = self.file_pairs[idx]

        if self.transform:
          data_item = self.transform(data_item)

        return data_item

keys = ['lr', 'hr'] # Keys from the get_data_dicts
transform = Compose([
    # Opens the path using nibabel and turns the file into a numpy array.
    LoadImageD(keys=keys),
    # Add the batch dimension at the begining [Batch, Channel, H, W, D]
    EnsureChannelFirstd(keys=keys),
    # Using the file metadata (loaded at LoadImageD), it re-orders the array's axes to match the axcodes.
    # RAS forces the axes to Right-to-Left, Anterior-to-Posterior, Superior-to-Inferior
    OrientationD(keys=keys, 
                 axcodes="RAS",
                 labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
    # Normalizes the intensities linearly to [0,1]
    ScaleIntensityRangeD(
        keys=keys, a_min=0, a_max=1000,
        b_min=0.0, b_max=1.0, clip=True
    ),
    # Make sure everything is getting the correct dimensions
    ResizeD(keys=keys, spatial_size=(512, 512, 128)),
    # To Tensor!
    ToTensorD(keys=keys)
])

def save_paired_slices(lr_vol, hr_vol, output_dir, prefix):
    """
    Slices 3D volumes along all 3 axes and saves pairs to disk.
    
    Args:
        lr_vol (np.array): Low Res 3D volume (H, W, D)
        hr_vol (np.array): High Res 3D volume (scaled H, W, D)
        output_dir (str): Root folder to save data
        prefix (str): Identifier for the specific volume (e.g., 'patient_001')
    """
    
    if torch.is_tensor(lr_vol): lr_vol = lr_vol.cpu().numpy()
    if torch.is_tensor(hr_vol): hr_vol = hr_vol.cpu().numpy()
    
    axes_map = {
        #'sagittal': 0, # Slicing from side to side
        #'coronal':  1, # Slicing from front to back
        'axial':    2  # Slicing from top to bottom
    }

    for axis_name, axis_idx in axes_map.items():
        save_path = os.path.join(output_dir, axis_name)
        os.makedirs(save_path, exist_ok=True)

        scale = hr_vol.shape[axis_idx] // lr_vol.shape[axis_idx]

        num_slices = lr_vol.shape[axis_idx]

        print(f"Processing {prefix} | {axis_name} | Scale {scale} | num slices {num_slices} | LR Shape: {lr_vol.shape} | HR Shape: {hr_vol.shape}")

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
    
    data_dir = Path('data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/Data')
    output_dir = Path("data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/processed")
    
    try:
        train_ds = PairedMRIDataset(root_dir=data_dir, transform=transform)
        print(len(train_ds), "paired scans found in the dataset.")

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

        for i, batch in enumerate(train_loader):
            lr_img_3d = batch['lr'][0, 0] 
            hr_img_3d = batch['hr'][0, 0]
            
            # Generate a unique name
            vol_name = f"vol_{i:03d}"
            
            # Run generation
            save_paired_slices(lr_img_3d, hr_img_3d, output_dir, vol_name)

        print(f"\nProcessing complete. Data saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")