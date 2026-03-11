import os
import numpy as np
from PIL import Image
import json
import glob
from tqdm import tqdm


def normalize_to_uint8(img_array):
    """
    Normalizes a float array to 0-255 uint8 range for PNG saving.
    """
    # 1. Handle dimensions (remove channel dim if present, e.g. 1x512x512 -> 512x512)
    if img_array.ndim == 3 and img_array.shape[0] == 1:
        img_array = img_array.squeeze(0)

    # 2. Normalize to 0-1 range first
    img_min = img_array.min()
    img_max = img_array.max()

    if img_max - img_min > 0:
        img_norm = (img_array - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_array)

    # 3. Convert to 0-255 uint8
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return img_uint8


if __name__ == "__main__":
    SOURCE = "./data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/processed/axial"
    DEST = "./data/Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research v3/train/axial"

    # Create directory structure
    images_dir_hr = os.path.join(DEST, "hr_images")
    images_dir_lr = os.path.join(DEST, "lr_images")
    os.makedirs(images_dir_hr, exist_ok=True)
    os.makedirs(images_dir_lr, exist_ok=True)

    metadata_path = os.path.join(DEST, "metadata.jsonl")

    files = glob.glob(os.path.join(SOURCE, "*.npz"))
    print(f"Found {len(files)} .npz files processing...")

    with open(metadata_path, "w") as json_file:
        for filepath in tqdm(files):
            try:
                # Load NPZ
                data = np.load(filepath)
                hr_arr = data["hr"]
                lr_arr = data["lr"]

                # Get Filename base
                base_name = os.path.splitext(os.path.basename(filepath))[0]

                # Normalize and Convert
                hr_img = Image.fromarray(normalize_to_uint8(hr_arr))
                lr_img = Image.fromarray(normalize_to_uint8(lr_arr))

                # Define save paths relative to the metadata file
                hr_rel_path = os.path.join("hr_images", f"{base_name}.png")
                lr_rel_path = os.path.join("lr_images", f"{base_name}.png")

                # Save Images
                hr_img.save(os.path.join(DEST, hr_rel_path))
                lr_img.save(os.path.join(DEST, lr_rel_path))

                # Write Metadata
                # We use keys "image", "conditioning_image", and "text" to match standard HF logic
                entry = {
                    "image": hr_rel_path,
                    "conditioning_image": lr_rel_path,
                    "text": "high quality mri scan",
                }
                json.dump(entry, json_file)
                json_file.write("\n")

            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    print(f"Conversion complete! Dataset saved to: {DEST}")
