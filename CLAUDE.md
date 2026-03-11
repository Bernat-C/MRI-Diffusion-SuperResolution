# CLAUDE.md — MRI-Diffusion-SuperResolution

## Project Goal
Apply PEFT methods (T2I Adapter, LoRA) to diffusion models for low-field → high-field MRI super-resolution.

## Repository Layout
```
MRIDiffusion/
  t2iadapter/
    utils.py          # All generate_* functions, encode_prompt, DC helpers
    MRIProjector.py   # MRIProjector, LatentMRIProjector, InverseProjectorLearned, robust_mri_scale
    t2iadapter.py     # T2I Adapter model
    config.py         # T2IConfig dataclass
  slicedMRI/
    dataset.py        # FastMRILazyDataset, PairedMRI_MiniDataset, PairedMRIDataset
    config.py         # DatasetConfig dataclass
    transform_to_2D_slices.py
  eval/
    eval.py
jupyter/              # Experiment notebooks
unused/               # Archived training scripts (ControlNet, LoRA)
```

## Key Models & Configs

### Base diffusion model
- **SD 1.5**: `sd-legacy/stable-diffusion-v1-5`
- Uses `encode_prompt_sd1x5` / `compute_embeddings_sd1x5` (NOT the SDXL variants)
- SD 1.5 UNet expects `encoder_hidden_states` shape `[B, 77, 768]`

### VAE
- Default: `microsoft/mri-autoencoder-v0.1` — **special handling required**:
  - After encoding: interpolate latents from 128×128 → 64×64
  - Before decoding: interpolate latents from 64×64 → 128×128
- Check with: `"microsoft/mri-autoencoder-v0.1" in vae.config.get("_name_or_path", "")`

### MRI Projector
- `MRIProjector`: 1×1 conv, maps 1-channel MRI → 3-channel RGB in `[-1, 1]` (for VAE/adapter input)
- `LatentMRIProjector`: maps MRI-VAE latents (4×128×128) → SD1.5 latents (4×64×64) via strided 1×1 conv
- `InverseProjectorLearned`: upsample + 2-conv block for inverse mapping

### T2IConfig defaults
- `resolution=512`, `mixed_precision="bf16"`, `use_8bit_adam=True`
- DDPM: `v_prediction`, `trailing` timestep spacing, zero-SNR betas rescaling
- `partial_start_step=800`

### DatasetConfig
- Dataset dict keys: `"lr"` (tensor `[1,H,W]`), `"hr"` (tensor `[1,H,W]`), `"txt"` (str)
- LR is simulated by Gaussian blur + bicubic downsample/upsample (scale_factor=4.0)
- FastMRI: filters by `contrast_filter` and `strength_filter`; subject-level train/val/test split
- PairedMRI_MiniDataset: 11 subjects × 24 axial FLAIR slices; uses `.npz` disk cache under `cache_dir`

## Generation Pipeline Functions (`utils.py`)

| Function | Description |
|---|---|
| `generate_mri_slices` | Full DDPM + T2I adapter (no partial init) |
| `generate_mri_slices_partial` | SDEEdit partial diffusion + T2I adapter + MRIProjector |
| `generate_mri_slices_partial_dc` | Partial diffusion + per-step latent-space DC + T2I adapter |
| `generate_mri_slices_partial_latent_align_dc_no_t2i` | Partial DC without T2I adapter; UNet input is `cat([latents_gen, latents_lr], dim=1)` → requires 8-channel UNet |
| `generate_mri_slices_partial_latent_align_dc` | Partial DC + T2I adapter + separate `vae_encoder`/`vae_decoder` + `LatentMRIProjector` |

## Data Consistency (DC)
- `apply_frequency_consistency_soft(latents_pred, latents_cond, reduction_factor, taper_width)`:
  Soft FFT-based low-pass replacement — replaces central frequencies of predicted latents with LR reference.
- `make_lowpass_mask(h, w, reduction_factor, device, taper_width)`: box-metric radial mask with raised-cosine taper.
- Applied at each denoising step using the **same** `noise_init` to keep noise alignment.
- Optional final pixel-space DC pass after decoding.

## Coding Conventions
- All tensors use `weight_dtype=torch.bfloat16` during inference; cast to float before VAE decode.
- `robust_mri_scale(tensor, pmin=0.5, pmax=99.5)`: percentile normalization → `[0, 1]`, used on all final outputs.
- `accelerator.device` is the canonical device reference throughout.
- `vae_scale` is always read via: `getattr(getattr(vae, "config", {}), "scaling_factor", None) or getattr(vae, "scaling_factor", 1.0)`

## Notebooks (jupyter/)
- `PartialDC_LatentProjector_no_finetune.ipynb` — latest; no T2I adapter, latent projector only
- `PartialDC_LatentProjector_T2I_adapter.ipynb` — latent projector + T2I adapter
- `PartialDC_T2I_adapter.ipynb` — partial diffusion + DC + T2I adapter
- `Partial_T2I_adapter.ipynb` — partial diffusion + T2I adapter (no DC)
- `T2I_adapter.ipynb` — baseline T2I adapter

## Known Issues
- `PairedMRIDataset.__init__` has a bug: `if not Path.exists()` should be `if not self.cache_dir.exists()`
