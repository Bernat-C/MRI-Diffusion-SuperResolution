# MRI Super-Resolution with Diffusion Models

This repository contains the implementation for MRI Super-Resolution (SR) using generative priors from large-scale diffusion models. The project focuses on adapting models pre-trained on natural images to the medical domain using Parameter-Efficient Fine-Tuning (PEFT).

## Project Structure

| Folder / File | Description |
| :--- | :--- |
| `notebooks/` | Contains the jupyter notebooks used to conduct the tests accross the project. |
| `slicedMRI/` | Contains the first implementation for a dataset for sliced MRIs. |
| `src/` | Source code of the project, including implementations for the adapters that have been tried, datasets and evaluation. |


---

### Installation
```bash
# 1. Create a virtual environment named 'mri-env'
# (You can replace 'mri-env' with any name you prefer)
python -m venv mri-env

# 2. Activate the environment
source mri-env/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

## Authors

- Bernat Comas i Machuca - Technical University of Munich

- Hannes Leonhard - Technical University of Munich