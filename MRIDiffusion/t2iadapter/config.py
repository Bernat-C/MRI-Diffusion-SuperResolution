from dataclasses import dataclass
from typing import Union


@dataclass
class T2IConfig:
    pretrained_model_name_or_path: str = "sd-legacy/stable-diffusion-v1-5"
    pretrained_vae_model_name_or_path: Union[str, None] = (
        "microsoft/mri-autoencoder-v0.1"
    )
    revision: Union[str, None] = None
    tokenizer_name: Union[str, None] = None
    output_dir: str = "./out/t2i_adapter"
    seed: int = 42
    resolution: int = 512
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    train_batch_size: int = 32
    test_batch_size: int = 16
    num_train_epochs: int = 200
    max_train_steps: int = 4000
    checkpointing_steps: int = 500
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    scale_lr: bool = False
    config: str = ""
    lr_scheduler_name: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    use_8bit_adam: bool = True
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    logging_dir: str = "./logs/t2i_adapter"
    allow_tf32: bool = False
    report_to: str = "wandb"
    media_reporting_step: int = 100
    mixed_precision: Union[str, None] = None
    enable_xformers_memory_efficient_attention: bool = False
    set_grads_to_none: bool = False
    proportion_empty_prompts: float = 0.1
    tracker_project_name: str = "mri_t2i_adapter_v1.5"
    ddpm_scheduler_prediction_type: str = "v_prediction"  # velocity prediction
    ddpm_scheduler_timestep_spacing: str = "trailing"  # for zero-SNR
    ddpm_scheduler_rescale_betas_zero_snr: bool = True  # enforces pure noise at t=1000
