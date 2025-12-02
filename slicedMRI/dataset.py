import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SlicedMRIDataset(Dataset):
    def __init__(self, root_dir, axes=['axial', 'sagittal', 'coronal'], transform=None):
        """
        Args:
            root_dir (str or Path): Path to the 'processed' folder.
            axes (list): Which anatomical axes to load. 
                         Defaults to all three. 
                         Use ['axial'] if you only want top-down slices.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.file_list = []

        # 1. Gather all .npz files from the specified subdirectories
        for axis in axes:
            axis_path = self.root_dir / axis
            if not axis_path.exists():
                print(f"Warning: Directory {axis_path} does not exist. Skipping.")
                continue
            
            # Find all .npz files in this axis folder
            files = list(axis_path.glob("*.npz"))
            self.file_list.extend(files)

        print(f"Dataset Initialized. Found {len(self.file_list)} slices across {axes}.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        data = np.load(file_path)
        
        lr_slice = data['lr'] # Shape: (H, W)
        hr_slice = data['hr'] # Shape: (Scaled_H, Scaled_W)
        
        lr_tensor = torch.from_numpy(lr_slice).float().unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_slice).float().unsqueeze(0)

        sample = {'lr': lr_tensor, 'hr': hr_tensor, 'path': str(file_path.name)}

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_batch(dataloader):
    """
    Visualizes a single batch from the loader.
    """
    batch = next(iter(dataloader))
    lr_imgs = batch['lr']
    hr_imgs = batch['hr']
    
    # Grab the first item in the batch
    lr = lr_imgs[0].squeeze() # remove channel dim for plotting
    hr = hr_imgs[0].squeeze()
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Low Res (Input)\nShape: {lr.shape}")
    plt.imshow(lr, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"High Res (Target)\nShape: {hr.shape}")
    plt.imshow(hr, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
if __name__ == "__main__":

    PROCESSED_DIR = Path("data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/processed")
    
    train_ds = SlicedMRIDataset(
        root_dir=PROCESSED_DIR, 
        axes=['axial']#, 'sagittal', 'coronal'] 
    )

    print(train_ds[0]['lr'].shape, train_ds[0]['hr'].shape)
    print(train_ds[1]['lr'].shape, train_ds[1]['hr'].shape)
    print(train_ds[2]['lr'].shape, train_ds[2]['hr'].shape)

    if len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        
        print(f"DataLoader ready with batch size 8.")
        
        try:
            show_batch(train_loader)
            
            print("\nSimulating one iteration...")
            batch = next(iter(train_loader))
            lr = batch['lr']
            hr = batch['hr']
            print(f"Batch LR Shape: {lr.shape}") # Should be [8, 1, H, W]
            print(f"Batch HR Shape: {hr.shape}") # Should be [8, 1, H_scale, W_scale]
            
        except Exception as e:
            print(f"Error during loading: {e}")
    else:
        print("No files found. Please check the PROCESSED_DIR path.")