import torch
import torch.nn.functional as F

from PIL import Image
import numpy as np

def get_res_shifting_latents(hr_latents, lr_latents, timesteps, scheduler):
    """
    Implements the Res-SRDiff shifting process.
    Moves from HR towards LR as t increases, with added variance.
    """
    # Get the alpha schedule from the noise scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(hr_latents.device)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    
    # Res-SRDiff Shifting: x_t = sqrt(alpha_t)*HR + (1-sqrt(alpha_t))*LR + noise
    # As t -> max, alpha_t -> 0, so x_t becomes LR + noise
    mu_t = (alpha_t ** 0.5) * hr_latents + (1 - (alpha_t ** 0.5)) * lr_latents
    
    # Variance scaling
    noise = torch.randn_like(hr_latents)
    sigma_t = (1 - alpha_t) ** 0.5
    
    return mu_t + sigma_t * noise

def prepare_condition_image(image, target_size=(512, 512)):
    """Ensures the LR image is the correct size and channel count for ControlNet."""
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    if image.shape[-2:] != target_size:
        image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)
    return image

@torch.no_grad()
def log_validation(unet, controlnet, vae, val_dataloader, noise_scheduler, weight_dtype, accelerator, fixed_embeds, num_inference_steps=20):
    unet.eval()
    controlnet.eval()

    # Get a single batch
    val_batch = next(iter(val_dataloader))
    hr_raw = val_batch['hr'][0:1].to(accelerator.device, dtype=weight_dtype)
    lr_raw = val_batch['lr'][0:1].to(accelerator.device, dtype=weight_dtype)
    
    # Prepare ControlNet condition (pixel space)
    control_image = prepare_condition_image(lr_raw) 
    
    # Encode LR to get starting point for Res-Shifting
    lr_input = lr_raw.expand(-1, 3, -1, -1) if lr_raw.shape[1] == 1 else lr_raw
    lr_latents_anchor = vae.encode(lr_input).latent_dist.sample() * vae.config.scaling_factor

    # Setup Scheduler
    noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    timesteps = noise_scheduler.timesteps

    # Initialize the "Shifted" state
    # x_T = sqrt(alpha_T)*HR + (1-sqrt(alpha_T))*LR + noise.
    latents = get_res_shifting_latents(lr_latents_anchor, lr_latents_anchor, timesteps[0], noise_scheduler)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)

    # Multi-step Denoising Loop
    for i, t in enumerate(timesteps):
        # Predict ControlNet residuals
        down_res, mid_res = controlnet(
            latents, t, 
            encoder_hidden_states=fixed_embeds[0:1], 
            controlnet_cond=control_image, 
            return_dict=False
        )

        # Predict HR latents (x0)
        model_pred = unet(
            latents, t, 
            encoder_hidden_states=fixed_embeds[0:1],
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res
        ).sample

        # 4. MANUAL RES-SRDIFF REVERSE STEP
        # Manually shift from the LR anchor toward the predicted HR.
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0).to(accelerator.device)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t_prev = alphas_cumprod[prev_t].view(-1, 1, 1, 1)

        # Transition math: x_{t-1} = sqrt(alpha_{t-1})*pred_x0 + (1-sqrt(alpha_{t-1}))*LR
        # We omit the additional variance (gamma) during inference for a cleaner SR result.
        latents = (alpha_t_prev ** 0.5) * model_pred + (1 - alpha_t_prev ** 0.5) * lr_latents_anchor

        if prev_t > 0:
            noise = torch.randn_like(latents)
            # Use standard DDPM variance logic or a fixed gamma as per the Res-SRDiff paper
            variance = ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) ** 0.5
            latents = latents + variance * noise

    # 5. Decode Final Result
    # Remember to reverse the [-1, 1] normalization in decode_to_vis
    gen_vis = decode_to_vis(latents, vae)
    hr_vis = decode_to_vis(hr_raw, vae, is_latent=False)
    lr_vis = decode_to_vis(lr_raw, vae, is_latent=False)

    combined = np.hstack([lr_vis, gen_vis, hr_vis])
    return Image.fromarray(combined)

def decode_to_vis(data, vae, is_latent=True):
    if is_latent:
        # Decode latent to pixel space
        decoded = vae.decode(data / vae.config.scaling_factor).sample
    else:
        decoded = data
        
    # Standardize to [0, 1] and move channels to the end: (B, C, H, W) -> (B, H, W, C)
    img = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    img_np = (img[0] * 255).astype(np.uint8)

    # If image is grayscale (H, W, 1), convert to (H, W, 3)
    if img_np.shape[-1] == 1:
        img_np = np.concatenate([img_np] * 3, axis=-1)
        
    return img_np

# PRE-COMPUTING THE PROMPT EMBEDS
def get_fixed_prompt_embeds(tokenizer, text_encoder, accelerator, prompt="medical mri scan, high resolution"):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length, truncation=True)
    inputs = inputs.to(accelerator.device)
    with torch.no_grad():
        prompt_embeds = text_encoder(inputs.input_ids)[0]
    return prompt_embeds