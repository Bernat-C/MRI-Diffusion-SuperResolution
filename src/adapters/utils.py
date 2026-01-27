from typing import Any

import random
import numpy as np
import torch
import accelerate
from transformers import PretrainedConfig

# GENERAL

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    # This reads the config file from the HF Hub or local path
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )

    # Identify the class name from the config (e.g., "CLIPTextModel")
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection

    # elif model_class == "RobertaSeriesModelWithTransformation":
    #     # Note: This is specific to AltDiffusion models
    #     from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
    #     return RobertaSeriesModelWithTransformation

    else:
        raise ValueError(f"{model_class} is not supported.")

def log_configs(config) -> dict[str, Any]:
    return {
        "data_dir": config.data_dir.__str__(),
        #"modality": config.modality,
        "slice_axis": config.slice_axis,
        "seed": config.seed,
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "pretrained_vae_model_name_or_path": config.pretrained_vae_model_name_or_path,
        "tokenizer_name": config.tokenizer_name,
        "resolution": config.resolution,
        "crops_coords_top_left_h": config.crops_coords_top_left_h,
        "crops_coords_top_left_w": config.crops_coords_top_left_w,
        "train_batch_size": config.train_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "max_train_steps": config.max_train_steps,
        "checkpointing_steps": config.checkpointing_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "scale_lr": config.scale_lr,
        "lr_scheduler_name": config.lr_scheduler_name,
        "lr_warmup_steps": config.lr_warmup_steps,
        "lr_num_cycles": config.lr_num_cycles,
        "lr_power": config.lr_power,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "adam_weight_decay": config.adam_weight_decay,
        "adam_epsilon": config.adam_epsilon,
        "max_grad_norm": config.max_grad_norm,
        "proportion_empty_prompts": config.proportion_empty_prompts,
        "ddpm_scheduler_prediction_type": config.ddpm_scheduler_prediction_type,
        "ddpm_scheduler_timestep_spacing": config.ddpm_scheduler_timestep_spacing,
        "ddpm_scheduler_rescale_betas_zero_snr": config.ddpm_scheduler_rescale_betas_zero_snr,
        "lora_alpha": getattr(config, "lora_alpha", None),
        "lora_rank": getattr(config, "lora_rank", None),
    }

# PROMPT ENCODING

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
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

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# Simplified encode_prompt for SD1.5 (replace your existing function)
def encode_prompt_sd1x5(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts: # CFG Dropout enabled here!
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
        )[0] # [0] extracts the prompt_embeds from the tuple
    return prompt_embeds # Shape: [B, 77, 768]


# Simplified compute_embeddings for SD1.5
def compute_embeddings_sd1x5(batch, proportion_empty_prompts, text_encoders, tokenizers, device, is_train=True):
    prompt_batch = batch['txt']

    # Now calls the simplified SD1.5 encoder
    prompt_embeds = encode_prompt_sd1x5(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )

    prompt_embeds = prompt_embeds.to(device)

    # SD1.5 UNet only needs the prompt_embeds tensor
    return {"prompt_embeds": prompt_embeds} # Return only the required embedding


# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, device, args, is_train=True):
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    prompt_batch = batch['txt']
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    )
    add_text_embeds = pooled_prompt_embeds
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return {"prompt_embeds": prompt_embeds}, unet_added_cond_kwargs