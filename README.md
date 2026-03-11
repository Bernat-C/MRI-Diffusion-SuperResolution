# Super-Resolution for MRI with Diffusion Models

This repository contains the implementation for our paper on MRI Super-Resolution (SR) using generative priors from large-scale diffusion models pre-trained on natural images. We investigate Parameter-Efficient Fine-Tuning (PEFT) methods (**T2I-Adapter**, **ControlNet**, **LoRA**, and training from scratch) to adapt Stable Diffusion 1.5 for enhancing low-field MRI scans to high-field quality. Additionally, we implement a **Spectral Predictor-Corrector (PC) inference** scheme with partial diffusion initialization to enforce anatomical fidelity and reduce hallucinations, and adapt the **ResDiff** architecture as a baseline.

## Key Findings

- PEFT methods alone struggle to bridge the domain gap between natural RGB images and single-channel MRI data.
- Training from scratch on MRI data significantly outperforms all PEFT approaches (PSNR: 19.01 vs. 8.53 for the best PEFT method, LoRA).
- Spectral PC inference is crucial for preserving anatomical structure, particularly for high-frequency details.
- Partial diffusion initialization benefits models trained from scratch on MRI data.

## Methods

### PEFT Conditioning Approaches
All methods condition a frozen Stable Diffusion 1.5 U-Net on low-field MRI input:

| Method | Description |
|:---|:---|
| **T2I-Adapter** | Lightweight auxiliary network injecting spatial conditioning into the U-Net |
| **ControlNet** | Trainable copy of frozen U-Net blocks connected via zero convolutions |
| **LoRA** | Low-rank parameterized update matrices applied to the frozen U-Net |
| **Scratch** | Full U-Net trained from scratch on MRI data (non-PEFT baseline) |

### Latent Alignment & Domain Adaptation
- **MRI Projector** (`MRIProjector`): 1x1 convolution mapping 1-channel MRI to 3-channel input for the VAE/adapter.
- **MRI-VAE** (`microsoft/mri-autoencoder-v0.1`): Domain-adapted VAE preserving anatomical structure (requires 128x128 <-> 64x64 latent interpolation).
- **Latent MRI Projector** (`LatentMRIProjector`): Aligns MRI-VAE latents with SD1.5 latent space via strided convolution.

### Spectral Predictor-Corrector Inference
- **Partial Diffusion Initialization (SDEdit)**: Start denoising from an intermediate timestep T' < T using the diffused low-field latent instead of pure Gaussian noise.
- **Spectral PC Step**: At each denoising step, replace low-frequency components of predicted latents with those from the LF reference using an FFT-based raised-cosine tapered mask.

### ResDiff Baseline
- CNN + diffusion model architecture for residual-based SR at 256x256 resolution.
- High-frequency guided cross-attention and adaptive high-pass filtering.

## Project Structure

```
MRI-Diffusion-SuperResolution/
├── src/                        # Source code
│   ├── t2iadapter/             # T2I Adapter, projectors, generation pipelines
│   │   ├── utils.py            # All generate_* functions, encode_prompt, DC helpers
│   │   ├── MRIProjector.py     # MRIProjector, LatentMRIProjector, InverseProjectorLearned
│   │   ├── t2iadapter.py       # T2I Adapter model architecture
│   │   └── config.py           # T2IConfig dataclass
│   ├── adapters/               # ControlNet, LoRA, ResDiff implementations
│   │   ├── res_srdiff.py       # ResDiff architecture
│   │   ├── modules.py          # Shared adapter modules
│   │   └── utils.py            # Adapter utilities
│   ├── slicedMRI/              # Dataset and MRI preprocessing
│   │   ├── dataset.py          # FastMRILazyDataset, PairedMRI_MiniDataset
│   │   ├── config.py           # DatasetConfig dataclass
│   │   ├── transform_to_2D_slices.py  # DICOM/NIfTI to 2D slice conversion
│   │   ├── generate_train.py   # Training data generation
│   │   └── view_utils.py       # Visualization utilities
│   ├── datasets/               # Additional dataset utilities
│   │   └── mri_datasets.py
│   └── eval/                   # Evaluation metrics (PSNR, SSIM, HFEN, NMSE)
│       └── eval.py
├── notebooks/                  # Jupyter notebooks (training, evaluation, visualization)
├── report/                     # LaTeX paper and figures
├── requirements.txt
└── README.md
```

## Notebooks

### Training
| Notebook | Description |
|:---|:---|
| [`PEFT Training T2I-Adapter.ipynb`](notebooks/PEFT%20Training%20T2I-Adapter.ipynb) | Train T2I-Adapter with full diffusion |
| [`PEFT Training T2I-Adapter PartialDiffusion.ipynb`](notebooks/PEFT%20Training%20T2I-Adapter%20PartialDiffusion.ipynb) | Train T2I-Adapter with partial diffusion initialization |
| [`PEFT Training ControlNet PartialDiffusion.ipynb`](notebooks/PEFT%20Training%20ControlNet%20PartialDiffusion.ipynb) | Train ControlNet with partial diffusion initialization |
| [`PEFT Training LoRA PartialDiffusion.ipynb`](notebooks/PEFT%20Training%20LoRA%20PartialDiffusion.ipynb) | Train LoRA with partial diffusion initialization |
| [`PEFT Training Scratch PartialDiffusion.ipynb`](notebooks/PEFT%20Training%20Scratch%20PartialDiffusion.ipynb) | Train U-Net from scratch with partial diffusion |
| [`PEFT Training Scratch.ipynb`](notebooks/PEFT%20Training%20Scratch.ipynb) | Train U-Net from scratch with full diffusion |
| [`ResDif_execution.ipynb`](notebooks/ResDif_execution.ipynb) | Train and run ResDiff baseline |

### Evaluation
| Notebook | Description |
|:---|:---|
| [`PEFT Evaluation Baseline.ipynb`](notebooks/PEFT%20Evaluation%20Baseline.ipynb) | Quantitative evaluation of all PEFT methods (PSNR, SSIM, HFEN, NMSE) |
| [`PEFT Evaluation PC-Inference.ipynb`](notebooks/PEFT%20Evaluation%20PC-Inference.ipynb) | Evaluate effect of spectral predictor-corrector inference |
| [`PEFT Evaluation Prompts.ipynb`](notebooks/PEFT%20Evaluation%20Prompts.ipynb) | Evaluate impact of different text prompts on generation quality |
| [`PEFT Qualitative Visualization.ipynb`](notebooks/PEFT%20Qualitative%20Visualization.ipynb) | Generate qualitative comparison figures |

### Data
| Notebook | Description |
|:---|:---|
| [`dataset.ipynb`](notebooks/dataset.ipynb) | Dataset exploration, preprocessing, and visualization |

## Dataset

We use the **fastMRI Brain DICOM** dataset (T2 modality, 3T field strength):
- 2,524 subject volumes split 80/10/10 (train/val/test)
- ~61k training slices, ~7.8k validation slices, ~7.8k test slices
- All slices zero-padded to 512x512 resolution

Low-field MRI is simulated via Gaussian blur + bicubic downsampling (factor 4x) and upsampling back to target resolution.

**Note**: The fastMRI dataset must be obtained separately from [https://fastmri.med.nyu.edu/](https://fastmri.med.nyu.edu/).

## Installation

```bash
# 1. Create a virtual environment
python -m venv mri-env

# 2. Activate the environment
source mri-env/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.x (with CUDA for GPU training)
- Hugging Face Transformers & Accelerate
- MONAI (medical imaging toolkit)
- An NVIDIA GPU with at least 40GB VRAM (A100 recommended) for training

**Note**: The `requirements.txt` lists `torch==2.9.1+cpu` — for GPU training, install PyTorch with CUDA support from [https://pytorch.org/](https://pytorch.org/) instead.

## Usage

1. **Prepare the dataset**: Download fastMRI Brain DICOM data and use `dataset.ipynb` to preprocess and convert to 2D slices.

2. **Train a model**: Open one of the training notebooks (e.g., `PEFT Training T2I-Adapter PartialDiffusion.ipynb`) and follow the cells. Training configuration is managed through `T2IConfig` in `src/t2iadapter/config.py`.

3. **Evaluate**: Use the evaluation notebooks to compute quantitative metrics or generate qualitative visualizations of the trained models.

All training notebooks are designed to run on a single GPU using `accelerate` for mixed-precision (bf16) training.

## Evaluation Metrics

| Metric | Description | Better |
|:---|:---|:---|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher |
| **SSIM** | Structural Similarity Index | Higher |
| **HFEN** | High-Frequency Error Norm | Lower |
| **NMSE** | Normalized Mean Squared Error | Lower |

## Authors

- **Bernat Comas i Machuca** — Technical University of Munich
- **Hannes Leonhard** — Technical University of Munich
