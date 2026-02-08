import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import PretrainedConfig
from typing import Dict, Union, Any
from accelerate import Accelerator
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from tqdm import tqdm

from MRIDiffusion.t2iadapter.config import T2IConfig
from MRIDiffusion.slicedMRI.config import DatasetConfig
from MRIDiffusion.t2iadapter.MRIProjector import robust_mri_scale


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts: float,
    is_train: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds: torch.Tensor = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds: torch.Tensor = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# Simplified encode_prompt for SD1.5 (replace your existing function)
def encode_prompt_sd1x5(
    prompt_batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts: float,
    is_train: bool = True,
) -> torch.Tensor:
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:  # CFG Dropout enabled here!
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    # SD1.5 only uses the first (and only) text encoder/tokenizer
    tokenizer = tokenizers[0]
    text_encoder = text_encoders[0]

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # SD1.5 uses the last hidden state of the standard CLIP Text Model
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            # SD1.5 does NOT need output_hidden_states=True
        )[
            0
        ]  # [0] extracts the prompt_embeds from the tuple
    return prompt_embeds  # Shape: [B, 77, 768]


# Simplified compute_embeddings for SD1.5 (replace your existing function)
def compute_embeddings_sd1x5(
    batch: Dict,
    proportion_empty_prompts: float,
    text_encoders,
    tokenizers,
    accelerator: Accelerator,
    is_train: bool = True,
) -> dict[str, torch.Tensor]:
    prompt_batch = batch["txt"]

    # Now calls the simplified SD1.5 encoder
    prompt_embeds = encode_prompt_sd1x5(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )

    prompt_embeds = prompt_embeds.to(accelerator.device)

    # SD1.5 UNet only needs the prompt_embeds tensor
    return {"prompt_embeds": prompt_embeds}  # Return only the required embedding


# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings(
    batch: Dict,
    args: T2IConfig,
    proportion_empty_prompts: float,
    text_encoders,
    tokenizers,
    accelerator: Accelerator,
    is_train: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    prompt_batch = batch["txt"]
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )
    add_text_embeds = pooled_prompt_embeds
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    prompt_embeds = prompt_embeds.to(accelerator.device)
    add_text_embeds = add_text_embeds.to(accelerator.device)
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return {"prompt_embeds": prompt_embeds}, unet_added_cond_kwargs


def _percentile_stretch(img: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-6):
    """Linearly stretch img so that pmin->0 and pmax->1. Works on single image (H,W,C) or (H,W)."""
    flat = img.flatten()
    vmin = np.percentile(flat, pmin)
    vmax = np.percentile(flat, pmax)
    if vmax - vmin < eps:
        return np.clip(img - vmin, 0.0, 1.0) * 0.0
    out = (img - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


from typing import Dict, Union, Any, List


def generate_mri_slices(
    batch: Dict[str, torch.Tensor],
    adapter: torch.nn.Module,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    prompt_embeds: dict[str, torch.Tensor],
    accelerator: Accelerator,
    num_inference_steps: int = 500,
    weight_dtype: torch.dtype = torch.float16,
    postprocess_mode: str = "percentile",  # options: "none", "percentile"
    pmin: float = 1.0,
    pmax: float = 99.0,
    gamma: Union[
        float, None
    ] = None,  # optional gamma correction (e.g. 0.9 or 1.1), None to skip
):
    """
    Performs diffusion inference and returns:
      - raw decoded images in numpy (B,H,W,C) in [0,1] (as produced by VAE decode & default scaling)
      - postprocessed images (same shape) according to postprocess_mode

    postprocess_mode:
      - "none": returns decoded images only (still clipped to [0,1] by default)
      - "percentile": apply per-image percentile stretch (pmin/pmax) to each generated image independently

    returns:
      - `image_batch_np` of shape (B, H, W, C)
      - `postprocessed` of shape (B, H, W, C)
    """
    device = accelerator.device
    bsz = batch["hr"].shape[0]
    h, w = batch["hr"].shape[-2:]
    if batch["lr"].ndim == 3:
        condition_sample = (
            batch["lr"].unsqueeze(1).to(device).float().expand(bsz, 3, h, w)
        )
    else:
        condition_sample = batch["lr"].to(device).float().expand(bsz, 3, h, w)
    adapter.eval()
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    with torch.no_grad():
        # adapter residuals
        down_block_additional_residuals = adapter(condition_sample)
        latent_shape = (bsz, unet.config.in_channels, h // 8, w // 8)
        latents_gen = torch.randn(latent_shape, device=device, dtype=weight_dtype)
        latents_gen = latents_gen * noise_scheduler.init_noise_sigma
        for t in noise_scheduler.timesteps:
            latent_model_input = noise_scheduler.scale_model_input(latents_gen, t)
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype)
                    for sample in down_block_additional_residuals
                ],
            ).sample
            latents_gen = noise_scheduler.step(noise_pred, t, latents_gen).prev_sample

        # decode latents -> images
        latents_gen = latents_gen.float() / vae.config.scaling_factor
        if vae.config["_name_or_path"] == "microsoft/mri-autoencoder-v0.1":
            latents_gen = F.interpolate(
                latents_gen, size=(128, 128), mode="bilinear", align_corners=False
            )
        image_batch = vae.decode(latents_gen).sample  # expected in [-1,1] for many VAEs
        # convert to [0,1]
        image_batch = (image_batch / 2.0 + 0.5).clamp(0.0, 1.0)

        # Move back to CPU: (B, C, H, W) -> (B, H, W, C)
        image_batch_np: np.ndarray = (
            image_batch.cpu().permute(0, 2, 3, 1).float().numpy()
        )
        # Prepare postprocessed copy
        postprocessed = np.empty_like(image_batch_np)

        for i in range(bsz):
            gen: np.ndarray = image_batch_np[i]  # (H,W,C) or (H,W,3)
            # if single channel in last dim, keep shape
            if gen.ndim == 3 and gen.shape[2] == 1:
                gen_img = np.squeeze(gen, axis=2)
            else:
                # for RGB, convert to grayscale for percentile reference (preserve RGB after scaling)
                gen_img = gen.mean(axis=2)

            if postprocess_mode == "none":
                proc: np.ndarray = gen.copy()
            elif postprocess_mode == "percentile":
                # stretch generated image itself
                if gen.ndim == 3 and gen.shape[2] > 1:
                    # scale each channel with the same factors
                    vmin = np.percentile(gen_img, pmin)
                    vmax = np.percentile(gen_img, pmax)
                    if vmax - vmin < 1e-6:
                        proc = np.clip(gen, 0.0, 1.0)
                    else:
                        proc = (gen - vmin) / (vmax - vmin)
                        proc = np.clip(proc, 0.0, 1.0)
                else:
                    proc = _percentile_stretch(gen, pmin=pmin, pmax=pmax)
                    if proc.ndim == 2:  # expand back to H,W,1
                        proc = proc[..., None]
            else:
                raise ValueError(f"Unknown postprocess_mode: {postprocess_mode}")

            # optional gamma
            if gamma is not None:
                proc = np.clip(proc, 0.0, 1.0) ** gamma
            # ensure shape (H,W,C) even for grayscale
            if proc.ndim == 2:
                proc = proc[..., None]
            postprocessed[i] = proc
        return image_batch_np, postprocessed


def generate_mri_slices_partial(
    batch: Dict[str, torch.Tensor],
    adapter: torch.nn.Module,
    mri_projector: torch.nn.Module,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    prompt_embeds: torch.Tensor,
    start_step: int,  # The new truncation parameter
    accelerator: Accelerator,
    num_inference_steps: int = 500,
    weight_dtype: torch.dtype = torch.float16,
):
    """
    Performs Partial Diffusion Inference:
    1. Encodes the LR image to latents (z_LR).
    2. Adds noise to z_LR to reach state t = start_step.
    3. Denoises from t = start_step -> 0.
    """
    device = accelerator.device
    bsz = batch["lr"].shape[0]
    mri_projector.eval()
    adapter.eval()
    with torch.no_grad():
        vae_input = mri_projector(batch["lr"].to(accelerator.device).float())
        down_block_additional_residuals = adapter(vae_input)
        model_name = vae.config.get("_name_or_path", "")
        is_special_vae = "microsoft/mri-autoencoder-v0.1" in model_name

        # Encode LR -> Latent LR
        latents_lr = vae.encode(vae_input.to(vae.dtype)).latent_dist.sample()
        latents_lr = latents_lr * vae.config.scaling_factor
        latents_lr = latents_lr.to(weight_dtype)
        # Special VAE downsampling fix (matches training loop logic)
        if is_special_vae:
            latents_lr = F.interpolate(
                latents_lr, size=(64, 64), mode="bilinear", align_corners=False
            )
        # Add Noise to reach t = start_step
        noise = torch.randn_like(latents_lr)
        timesteps_start = torch.full(
            (bsz,), start_step, device=device, dtype=torch.long
        )

        # This is our new starting point: Noisy LR Latents
        latents_gen = noise_scheduler.add_noise(latents_lr, noise, timesteps_start)
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        # Filter timesteps to only run from start_step -> 0
        inference_timesteps = [t for t in noise_scheduler.timesteps if t <= start_step]
        inference_timesteps = torch.tensor(inference_timesteps, device=device)
        for t in tqdm(
            inference_timesteps,
            disable=not accelerator.is_local_main_process,
            leave=False,
        ):
            latent_model_input = noise_scheduler.scale_model_input(latents_gen, t)
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype)
                    for sample in down_block_additional_residuals
                ],
            ).sample

            latents_gen = noise_scheduler.step(noise_pred, t, latents_gen).prev_sample
        latents_gen = latents_gen.float() / vae.config.scaling_factor
        if is_special_vae:
            latents_gen = F.interpolate(
                latents_gen, size=(128, 128), mode="bilinear", align_corners=False
            )
        image_batch_rgb = vae.decode(latents_gen.to(vae.dtype)).sample
        image_batch_grey = image_batch_rgb.mean(dim=1, keepdim=True)
        image_batch = robust_mri_scale(image_batch_grey)
        image_batch_np = image_batch.cpu().permute(0, 2, 3, 1).float().numpy()
        return image_batch_np, None


def make_lowpass_mask(
    h: int,
    w: int,
    reduction_factor: float,
    device: torch.device,
    taper_width: float = 0.4,
):
    cy, cx = h // 2, w // 2
    half_h = max(1, int(h / (2.0 * reduction_factor)))
    half_w = max(1, int(w / (2.0 * reduction_factor)))

    y = torch.arange(h, device=device) - cy
    x = torch.arange(w, device=device) - cx
    yy = y.abs().unsqueeze(1) / float(half_h)  # h x 1
    xx = x.abs().unsqueeze(0) / float(half_w)  # 1 x w
    radial = torch.max(yy, xx)  # box metric

    # map to [0,1], where <=1 inside low-pass
    mask = torch.clamp(1.0 - radial, 0.0, 1.0)

    if taper_width is not None and taper_width > 0.0:
        inner = 1.0 - taper_width
        outer = 1.0
        taper = (mask - inner) / (outer - inner + 1e-12)
        taper = torch.clamp(taper, 0.0, 1.0)
        # raised cosine on taper fraction
        mask_cos = 0.5 * (1.0 + torch.cos(torch.clamp(taper, 0.0, 1.0) * torch.pi))
        # where taper >=1 (fully inside) -> 1.0, where <=0 -> 0.0
        mask = torch.where(taper >= 1.0, torch.tensor(1.0, device=device), mask_cos)
        mask = torch.where(taper <= 0.0, torch.tensor(0.0, device=device), mask)
    else:
        mask = (mask > 0.0).float()

    return mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,h,w)


def apply_frequency_consistency_soft(
    latents_pred: torch.Tensor,
    latents_cond: torch.Tensor,
    reduction_factor: float = 4.0,
    taper_width: float = 0.12,
):
    """
    Softly replace low-freq bands of latents_pred with latents_cond using a smooth mask.
    Inputs: (B,C,H,W) real tensors. Returns real tensor (B,C,H,W).
    """
    B, C, H, W = latents_pred.shape
    # FFT (complex)
    fft_pred: torch.Tensor = torch.fft.fftshift(
        torch.fft.fft2(latents_pred.float(), norm="ortho"), dim=(-2, -1)
    )
    fft_cond: torch.Tensor = torch.fft.fftshift(
        torch.fft.fft2(latents_cond.float(), norm="ortho"), dim=(-2, -1)
    )
    mask = make_lowpass_mask(
        h=H,
        w=W,
        reduction_factor=reduction_factor,
        device=latents_pred.device,
        taper_width=taper_width,
    )  # (1,1,H,W)
    mask = mask.to(dtype=fft_pred.real.dtype)
    mask = mask.expand(B, C, H, W)
    # Complex blend
    fft_comb = fft_pred * (1.0 - mask) + fft_cond * mask
    # inverse FFT
    fft_ishift: torch.Tensor = torch.fft.ifftshift(fft_comb, dim=(-2, -1))
    latents_corr: torch.Tensor = torch.fft.ifft2(fft_ishift, norm="ortho")
    # return real part (vae latents are real)
    return latents_corr.real


def generate_mri_slices_partial_dc(
    batch: Dict[str, torch.Tensor],
    adapter: torch.nn.Module,
    mri_projector: torch.nn.Module,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    prompt_embeds: torch.Tensor,
    start_step: int,
    accelerator: Accelerator,
    num_inference_steps: int = 500,
    weight_dtype: torch.dtype = torch.float16,
    use_data_consistency: bool = True,
    dc_reduction_factor: float = 1.5,
    taper: float = 0.45,
    apply_final_pixel_dc: bool = True,
    vae_scale: float = None,
):
    """
    Generate MRI slices using partial diffusion (SDEEdit-style init) and latent-space DC.

    Parameters
    ----------
    batch : dict
        Must contain at least "lr" (low-res image tensor shape [B, C, H, W]) and optionally other metadata.
    adapter : nn.Module
        T2I Adapter used for conditioning (used here to produce additional residuals for the UNet).
    mri_projector: callable
        Function that maps the original LR batch['lr'] into a 3-channel RGB (or model input) space for the adapter/vae.
    unet : nn.Module
        The U-Net (diffusion model) used for denoising; expected HF diffusers-like signature.
    vae : nn.Module
        VAE with encode/decode. Expected interfaces:
         - vae.encode(img).latent_dist.sample()  (or adapt if deterministic)
         - vae.decode(latents) returning object with .sample or a tensor
    noise_scheduler : object
        The noise scheduler used for sampling. Expected to provide:
         - set_timesteps(num_inference_steps, device=device)
         - timesteps (iterable tensor)
         - add_noise(x0, noise, timesteps)
         - scale_model_input(x, t)
         - step(pred, t, sample) -> returns object with .prev_sample
    prompt_embeds : tensor
        Conditioning textual/image embeddings passed to UNet.
    start_step : int
        The starting (noisy) timestep to initialize from (SDEEdit skip).
    accelerator : optional (HF accelerate) - used to get device & local process checks
    use_data_consistency : bool
        Whether to apply latent DC at each step (recommended True).
    dc_reduction_factor : float
        How much of the central frequencies to preserve from the LR reference (e.g., 4.0 for 4x).
    taper : float
        Soft-edge width for the lowpass mask (0..0.5); higher -> smoother blend.
    apply_final_pixel_dc : bool
        Whether to apply final pixel-space DC on decoded image (one pass).
    vae_scale : float | None
        If your VAE uses a latent scaling factor (LDM style e.g. 0.18215), pass it here. If None, attempt to read from vae.config.

    Returns
    -------
    final_image_np : numpy array of shape (B, H, W, 1) or (B, H, W, C)
    None
        (second return kept for API compatibility)
    """

    device = (
        accelerator.device
        if accelerator is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_name = vae.config.get("_name_or_path", "")
    is_special_vae = "microsoft/mri-autoencoder-v0.1" in model_name
    mri_projector.eval()
    adapter.eval()
    with torch.no_grad():
        # Prepare conditioning RGB (projector may do channel conversion / scaling)
        condition_rgb = mri_projector(batch["lr"].to(device).float())
        down_block_res = adapter(condition_rgb)
        # Encode LR (clean latent) -- adapt to your VAE API
        enc = vae.encode(condition_rgb.to(vae.dtype))
        latents_lr_clean = (
            enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc
        )
    if vae_scale is None:
        vae_scale = getattr(
            getattr(vae, "config", {}), "scaling_factor", None
        ) or getattr(vae, "scaling_factor", 1.0)
    latents_lr_clean = latents_lr_clean * vae_scale
    latents_lr_clean = latents_lr_clean.to(weight_dtype)
    if is_special_vae:
        latents_lr_clean = F.interpolate(
            latents_lr_clean, size=(64, 64), mode="bilinear", align_corners=False
        )
    # Initialize noise
    noise_init = torch.randn_like(latents_lr_clean, device=device)
    # SDEEdit-style init: add noise at start_step using the SAME noise_init
    bsz = latents_lr_clean.shape[0]
    timesteps_start = torch.full(
        (bsz,), int(start_step), device=device, dtype=torch.long
    )
    latents_gen = noise_scheduler.add_noise(
        latents_lr_clean, noise_init, timesteps_start
    )
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    # Only keep timesteps that are <= start_step (we will denoise from start_step downwards)
    inference_timesteps = [t for t in noise_scheduler.timesteps if t <= start_step]
    inference_timesteps = torch.tensor(inference_timesteps, device=device)
    for t in tqdm(
        inference_timesteps,
        disable=True,
        leave=False,
    ):
        # scale model input (scheduler-specific)
        latent_model_input = noise_scheduler.scale_model_input(latents_gen, t)
        with torch.no_grad():
            unet_out = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=[
                    s.to(weight_dtype) for s in down_block_res
                ],
            )
            noise_pred = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        step_output = noise_scheduler.step(noise_pred, t, latents_gen)
        latents_gen = step_output.prev_sample
        # Apply latent-space data consistency (soft replacement) if requested
        if use_data_consistency:
            # compute noisy-lr at the same timestep using the SAME noise_init
            ts_cur = torch.full(
                (bsz,),
                int(t.item()) if isinstance(t, torch.Tensor) else int(t),
                device=device,
                dtype=torch.long,
            )
            noisy_lr_at_t = noise_scheduler.add_noise(
                latents_lr_clean, noise_init, ts_cur
            )
            # soft frequency replacement on *noisy* latents (keeps noise alignment)
            latents_gen = apply_frequency_consistency_soft(
                latents_gen,
                noisy_lr_at_t,
                reduction_factor=dc_reduction_factor,
                taper_width=taper,
            )
    latents_to_decode = latents_gen.float() / float(vae_scale)
    if is_special_vae:
        latents_to_decode = F.interpolate(
            latents_to_decode, size=(128, 128), mode="bilinear", align_corners=False
        )
    with torch.no_grad():
        decoded = vae.decode(latents_to_decode.to(vae.dtype))
        decoded_rgb = (
            decoded.sample if hasattr(decoded, "sample") else decoded
        )  # (B, C, H_img, W_img)
    # Optional final pixel-space DC (single hard/soft pass)
    if use_data_consistency and apply_final_pixel_dc:
        # collapse to single-channel; if multi-channel keep average
        decoded_gray = decoded_rgb.mean(dim=1, keepdim=True)  # (B,1,H,W)
        target_lr = batch["lr"].to(device).float()
        if target_lr.ndim == 3:
            target_lr = target_lr.unsqueeze(1)  # (B,1,H_lr,W_lr)
        # If sizes don't match, up/downsample target_lr to decoded_gray resolution
        if target_lr.shape[-2:] != decoded_gray.shape[-2:]:
            # using bilinear to match spatial dims
            target_lr_resized = F.interpolate(
                target_lr,
                size=decoded_gray.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            target_lr_resized = target_lr
        # final pixel-space DC using the same soft frequency helper
        final_gray = apply_frequency_consistency_soft(
            decoded_gray,
            target_lr_resized,
            reduction_factor=dc_reduction_factor,
            taper_width=taper,
        )
    else:
        final_gray = decoded_rgb.mean(dim=1, keepdim=True)
    image_batch = robust_mri_scale(final_gray)
    image_batch_np = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    return image_batch_np, None


def generate_mri_slices_partial_latent_align_dc_no_t2i(
    batch: Dict[str, torch.Tensor],
    mri_projector: torch.nn.Module,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    prompt_embeds: torch.Tensor,
    start_step: int,
    accelerator: Accelerator,
    num_inference_steps: int = 500,
    weight_dtype: torch.dtype = torch.float16,
    use_data_consistency: bool = True,
    dc_reduction_factor: float = 1.5,
    taper: float = 0.45,
    apply_final_pixel_dc: bool = True,
    vae_scale: float = None,
):
    device = (
        accelerator.device
        if accelerator is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # is_special_vae = "microsoft/mri-autoencoder-v0.1" in model_name
    mri_projector.eval()
    unet.eval()
    with torch.no_grad():
        # Prepare conditioning RGB (projector may do channel conversion / scaling)
        condition_rgb = mri_projector(batch["lr"].to(device).float())
        # Encode LR (clean latent) -- adapt to your VAE API
        enc = vae.encode(condition_rgb.to(vae.dtype))
        latents_lr_clean = (
            enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc
        )
    if vae_scale is None:
        vae_scale = getattr(
            getattr(vae, "config", {}), "scaling_factor", None
        ) or getattr(vae, "scaling_factor", 1.0)
    latents_lr_clean: torch.Tensor = latents_lr_clean * vae_scale
    latents_lr_clean = latents_lr_clean.to(weight_dtype)
    # Initialize noise
    noise_init = torch.randn_like(latents_lr_clean, device=device)
    # SDEEdit-style init: add noise at start_step using the SAME noise_init
    bsz = latents_lr_clean.shape[0]
    timesteps_start = torch.full(
        (bsz,), int(start_step), device=device, dtype=torch.long
    )
    latents_gen = noise_scheduler.add_noise(
        latents_lr_clean, noise_init, timesteps_start
    )
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    # Only keep timesteps that are <= start_step (we will denoise from start_step downwards)
    inference_timesteps = [t for t in noise_scheduler.timesteps if t <= start_step]
    inference_timesteps = torch.tensor(inference_timesteps, device=device)
    for i, t in tqdm(
        enumerate(inference_timesteps),
        total=len(inference_timesteps),
        disable=True,
        leave=False,
    ):
        latent_model_input = torch.cat([latents_gen, latents_lr_clean], dim=1)
        # scale model input (scheduler-specific)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            unet_out = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            )
            noise_pred = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        step_output = noise_scheduler.step(noise_pred, t, latents_gen)
        latents_gen = step_output.prev_sample
        # Apply latent-space data consistency (soft replacement) if requested
        if use_data_consistency:
            if i + 1 < len(inference_timesteps):
                next_t = inference_timesteps[i + 1]
                ts_target = torch.full(
                    (bsz,),
                    int(next_t),
                    device=device,
                    dtype=torch.long,
                )
                noisy_lr_target = noise_scheduler.add_noise(
                    latents_lr_clean, noise_init, ts_target
                )
                # soft frequency replacement on *noisy* latents (keeps noise alignment)
                latents_gen = apply_frequency_consistency_soft(
                    latents_gen,
                    noisy_lr_target,
                    reduction_factor=dc_reduction_factor,
                    taper_width=taper,
                )
    vae_decoding_scale = getattr(
        getattr(vae, "config", {}), "scaling_factor", None
    ) or getattr(vae)
    latents_to_decode = latents_gen.float() / float(vae_decoding_scale)
    with torch.no_grad():
        decoded = vae.decode(latents_to_decode.to(vae.dtype))
        decoded_rgb = (
            decoded.sample if hasattr(decoded, "sample") else decoded
        )  # (B, C, H_img, W_img)
    # Optional final pixel-space DC (single hard/soft pass)
    if use_data_consistency and apply_final_pixel_dc:
        # collapse to single-channel; if multi-channel keep average
        decoded_gray = decoded_rgb.mean(dim=1, keepdim=True)  # (B,1,H,W)
        target_lr = batch["lr"].to(device).float()
        if target_lr.ndim == 3:
            target_lr = target_lr.unsqueeze(1)  # (B,1,H_lr,W_lr)
        # If sizes don't match, up/downsample target_lr to decoded_gray resolution
        if target_lr.shape[-2:] != decoded_gray.shape[-2:]:
            # using bilinear to match spatial dims
            target_lr_resized = F.interpolate(
                target_lr,
                size=decoded_gray.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            target_lr_resized = target_lr
        # final pixel-space DC using the same soft frequency helper
        final_gray = apply_frequency_consistency_soft(
            decoded_gray,
            target_lr_resized,
            reduction_factor=dc_reduction_factor,
            taper_width=taper,
        )
    else:
        final_gray = decoded_rgb.mean(dim=1, keepdim=True)
    image_batch = robust_mri_scale(final_gray)
    image_batch_np = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    return image_batch_np, None


def generate_mri_slices_partial_latent_align_dc(
    batch: Dict[str, torch.Tensor],
    adapter: torch.nn.Module,
    mri_projector: torch.nn.Module,
    latent_projector: torch.nn.Module,
    unet: UNet2DConditionModel,
    vae_encoder: AutoencoderKL,
    vae_decoder: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    prompt_embeds: torch.Tensor,
    start_step: int,
    accelerator: Accelerator,
    num_inference_steps: int = 500,
    weight_dtype: torch.dtype = torch.float16,
    use_data_consistency: bool = True,
    dc_reduction_factor: float = 1.5,
    taper: float = 0.45,
    apply_final_pixel_dc: bool = True,
    vae_scale: float = None,
):
    """
    Generate MRI slices using partial diffusion (SDEEdit-style init) and latent-space DC.

    Parameters
    ----------
    batch : dict
        Must contain at least "lr" (low-res image tensor shape [B, C, H, W]) and optionally other metadata.
    adapter : nn.Module
        T2I Adapter used for conditioning (used here to produce additional residuals for the UNet).
    mri_projector: callable
        Function that maps the original LR batch['lr'] into a 3-channel RGB (or model input) space for the adapter/vae.
    latent_projector: nn.Module
        Function that maps a latent generated by the MRI vae to a latent space known by the SD1.5
    unet : nn.Module
        The U-Net (diffusion model) used for denoising; expected HF diffusers-like signature.
    vae : nn.Module
        VAE with encode/decode. Expected interfaces:
         - vae.encode(img).latent_dist.sample()  (or adapt if deterministic)
         - vae.decode(latents) returning object with .sample or a tensor
    noise_scheduler : object
        The noise scheduler used for sampling. Expected to provide:
         - set_timesteps(num_inference_steps, device=device)
         - timesteps (iterable tensor)
         - add_noise(x0, noise, timesteps)
         - scale_model_input(x, t)
         - step(pred, t, sample) -> returns object with .prev_sample
    prompt_embeds : tensor
        Conditioning textual/image embeddings passed to UNet.
    start_step : int
        The starting (noisy) timestep to initialize from (SDEEdit skip).
    accelerator : optional (HF accelerate) - used to get device & local process checks
    use_data_consistency : bool
        Whether to apply latent DC at each step (recommended True).
    dc_reduction_factor : float
        How much of the central frequencies to preserve from the LR reference (e.g., 4.0 for 4x).
    taper : float
        Soft-edge width for the lowpass mask (0..0.5); higher -> smoother blend.
    apply_final_pixel_dc : bool
        Whether to apply final pixel-space DC on decoded image (one pass).
    vae_scale : float | None
        If your VAE uses a latent scaling factor (LDM style e.g. 0.18215), pass it here. If None, attempt to read from vae.config.

    Returns
    -------
    final_image_np : numpy array of shape (B, H, W, 1) or (B, H, W, C)
    None
        (second return kept for API compatibility)
    """

    device = (
        accelerator.device
        if accelerator is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # is_special_vae = "microsoft/mri-autoencoder-v0.1" in model_name
    mri_projector.eval()
    adapter.eval()
    latent_projector.eval()
    with torch.no_grad():
        # Prepare conditioning RGB (projector may do channel conversion / scaling)
        condition_rgb = mri_projector(batch["lr"].to(device).float())
        down_block_res = adapter(condition_rgb)
        # Encode LR (clean latent) -- adapt to your VAE API
        enc = vae_encoder.encode(condition_rgb.to(vae_encoder.dtype))
        latents_lr_clean = (
            enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc
        )
    if vae_scale is None:
        vae_scale = getattr(
            getattr(vae_encoder, "config", {}), "scaling_factor", None
        ) or getattr(vae_encoder, "scaling_factor", 1.0)
    latents_lr_clean: torch.Tensor = latents_lr_clean * vae_scale
    latents_lr_clean = latents_lr_clean.to(weight_dtype)
    with torch.no_grad():
        latents_lr_clean = latent_projector(latents_lr_clean)
    # Initialize noise
    noise_init = torch.randn_like(latents_lr_clean, device=device)
    # SDEEdit-style init: add noise at start_step using the SAME noise_init
    bsz = latents_lr_clean.shape[0]
    timesteps_start = torch.full(
        (bsz,), int(start_step), device=device, dtype=torch.long
    )
    latents_gen = noise_scheduler.add_noise(
        latents_lr_clean, noise_init, timesteps_start
    )
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    # Only keep timesteps that are <= start_step (we will denoise from start_step downwards)
    inference_timesteps = [t for t in noise_scheduler.timesteps if t <= start_step]
    inference_timesteps = torch.tensor(inference_timesteps, device=device)
    for t in tqdm(
        inference_timesteps,
        disable=(accelerator is not None and not accelerator.is_local_main_process),
        leave=False,
    ):
        # scale model input (scheduler-specific)
        latent_model_input = noise_scheduler.scale_model_input(latents_gen, t)
        with torch.no_grad():
            unet_out = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=[
                    s.to(weight_dtype) for s in down_block_res
                ],
            )
            noise_pred = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        step_output = noise_scheduler.step(noise_pred, t, latents_gen)
        latents_gen = step_output.prev_sample
        # Apply latent-space data consistency (soft replacement) if requested
        if use_data_consistency:
            # compute noisy-lr at the same timestep using the SAME noise_init
            ts_cur = torch.full(
                (bsz,),
                int(t.item()) if isinstance(t, torch.Tensor) else int(t),
                device=device,
                dtype=torch.long,
            )
            noisy_lr_at_t = noise_scheduler.add_noise(
                latents_lr_clean, noise_init, ts_cur
            )
            # soft frequency replacement on *noisy* latents (keeps noise alignment)
            latents_gen = apply_frequency_consistency_soft(
                latents_gen,
                noisy_lr_at_t,
                reduction_factor=dc_reduction_factor,
                taper_width=taper,
            )
    vae_decoding_scale = getattr(
        getattr(vae_decoder, "config", {}), "scaling_factor", None
    ) or getattr(vae_decoder, "scaling_factor", 1.0)
    latents_to_decode = latents_gen.float() / float(vae_decoding_scale)
    with torch.no_grad():
        decoded = vae_decoder.decode(latents_to_decode.to(vae_decoder.dtype))
        decoded_rgb = (
            decoded.sample if hasattr(decoded, "sample") else decoded
        )  # (B, C, H_img, W_img)
    # Optional final pixel-space DC (single hard/soft pass)
    if use_data_consistency and apply_final_pixel_dc:
        # collapse to single-channel; if multi-channel keep average
        decoded_gray = decoded_rgb.mean(dim=1, keepdim=True)  # (B,1,H,W)
        target_lr = batch["lr"].to(device).float()
        if target_lr.ndim == 3:
            target_lr = target_lr.unsqueeze(1)  # (B,1,H_lr,W_lr)
        # If sizes don't match, up/downsample target_lr to decoded_gray resolution
        if target_lr.shape[-2:] != decoded_gray.shape[-2:]:
            # using bilinear to match spatial dims
            target_lr_resized = F.interpolate(
                target_lr,
                size=decoded_gray.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            target_lr_resized = target_lr
        # final pixel-space DC using the same soft frequency helper
        final_gray = apply_frequency_consistency_soft(
            decoded_gray,
            target_lr_resized,
            reduction_factor=dc_reduction_factor,
            taper_width=taper,
        )
    else:
        final_gray = decoded_rgb.mean(dim=1, keepdim=True)
    image_batch = robust_mri_scale(final_gray)
    image_batch_np = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    return image_batch_np, None


def plot_generated_and_ground_truth(
    generated_slices_np: np.ndarray,
    batch: Dict[str, torch.Tensor],
    postprocessed: np.ndarray = None,
    num_images_to_show: int = 4,
):
    """
    - generated_slices_np: raw decoded images (B,H,W,C)
    - postprocessed: optional array (B,H,W,C) of postprocessed for display
    """
    hr_slices_np = batch["hr"].cpu().numpy()
    if hr_slices_np.shape[1] == 1:
        hr_slices_np = np.squeeze(hr_slices_np, axis=1)  # -> (B, H, W)
    lr_slices_np = batch["lr"].cpu().numpy()
    if lr_slices_np.shape[1] == 1:
        lr_slices_np = np.squeeze(lr_slices_np, axis=1)  # -> (B, H, W)

    if generated_slices_np.ndim == 4 and generated_slices_np.shape[3] == 1:
        gen_raw = np.squeeze(generated_slices_np, axis=3)  # (B,H,W)
    else:
        gen_raw = generated_slices_np

    if postprocessed is None:
        gen_vis = gen_raw.copy()
    else:
        gen_vis = postprocessed

    batch_size = gen_vis.shape[0]
    num_plots = min(batch_size, num_images_to_show)
    axes: Union[np.ndarray, Axes]
    _, axes = plt.subplots(
        nrows=num_plots, ncols=4, figsize=(14, num_plots * 4), dpi=100
    )
    if num_plots == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_plots):
        # raw gen
        ax: Axes = axes[i, 0]
        if gen_raw.ndim == 3:  # grayscale (B,H,W)
            ax.imshow(gen_raw[i], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(gen_raw[i])
        ax.set_title(f"Generated RAW {i+1}")
        ax.axis("off")

        # postprocessed gen
        ax = axes[i, 1]
        if gen_vis.ndim == 3:
            ax.imshow(gen_vis[i], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(gen_vis[i])
        ax.set_title(f"Generated POST {i+1}")
        ax.axis("off")

        # GT
        ax = axes[i, 2]
        ax.imshow(hr_slices_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"GT HR {i+1}")
        ax.axis("off")

        # LR
        ax = axes[i, 3]
        ax.imshow(lr_slices_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"LR {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def log_configs(t2i_config: T2IConfig, dataset_config: DatasetConfig) -> dict[str, Any]:
    return {
        "data_dir": dataset_config.data_dir.__str__(),
        "modality": dataset_config.modality,
        "slice_axis": dataset_config.slice_axis,
        "seed": dataset_config.seed,
        "pretrained_model_name_or_path": t2i_config.pretrained_model_name_or_path,
        "pretrained_vae_model_name_or_path": t2i_config.pretrained_vae_model_name_or_path,
        "tokenizer_name": t2i_config.tokenizer_name,
        "seed_t2i": t2i_config.seed,
        "resolution": t2i_config.resolution,
        "crops_coords_top_left_h": t2i_config.crops_coords_top_left_h,
        "crops_coords_top_left_w": t2i_config.crops_coords_top_left_w,
        "train_batch_size": t2i_config.train_batch_size,
        "num_train_epochs": t2i_config.num_train_epochs,
        "max_train_steps": t2i_config.max_train_steps,
        "gradient_accumulation_steps": t2i_config.gradient_accumulation_steps,
        "learning_rate": t2i_config.learning_rate,
        "scale_lr": t2i_config.scale_lr,
        "lr_scheduler_name": t2i_config.lr_scheduler_name,
        "lr_warmup_steps": t2i_config.lr_warmup_steps,
        "lr_num_cycles": t2i_config.lr_num_cycles,
        "lr_power": t2i_config.lr_power,
        "adam_beta1": t2i_config.adam_beta1,
        "adam_beta2": t2i_config.adam_beta2,
        "adam_weight_decay": t2i_config.adam_weight_decay,
        "adam_epsilon": t2i_config.adam_epsilon,
        "max_grad_norm": t2i_config.max_grad_norm,
        "proportion_empty_prompts": t2i_config.proportion_empty_prompts,
        "ddpm_scheduler_prediction_type": t2i_config.ddpm_scheduler_prediction_type,
        "ddpm_scheduler_timestep_spacing": t2i_config.ddpm_scheduler_timestep_spacing,
        "ddpm_scheduler_rescale_betas_zero_snr": t2i_config.ddpm_scheduler_rescale_betas_zero_snr,
    }


def print_trainable_parameters(model: torch.nn.Module, name: str = "Model"):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"{name} || trainable params: {trainable_params:,d} || all params: {all_params:,d} || "
        f"size: {all_params * 4 / (1024**2):.2f} MB"
    )
