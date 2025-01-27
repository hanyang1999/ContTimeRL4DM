import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from huggingface_hub import hf_hub_download
from torchvision import transforms
from typing import Union, List
from PIL import Image
from scripts.src.ImageReward_VN import ImageRewardValue
from accelerate.logging import get_logger
from torch.optim.lr_scheduler import LambdaLR

logger = get_logger(__name__)

_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
}

def available_models() -> List[str]:
    """Returns the names of available ImageReward models"""
    return list(_MODELS.keys())

def ImageReward_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target

def load(name: str = "ImageReward-v1.0", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
    """Load a ImageReward model with dual BLIP architecture"""
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('Loading checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    if med_config is None:
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", 
                                        download_root or os.path.expanduser("~/.cache/ImageReward"))
    
    model = ImageRewardValue(med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)

    model.denoised_blip.load_state_dict(model.blip.state_dict())
    model.denoised_mlp.load_state_dict(model.mlp.state_dict())
    model.mlp.eval()
    
    model.freeze_original_models()
    print("Checkpoint loaded and original models frozen")
    model.freeze_denoised_layers(fix_rate=0.7)

    return model

def load_VN(name: str = "ImageReward-v1.0", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
    """Load a ImageReward model with dual BLIP architecture"""
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('Loading checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    if med_config is None:
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", 
                                        download_root or os.path.expanduser("~/.cache/ImageReward"))
    
    model = ImageRewardValue(med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)

    model.denoised_blip.load_state_dict(model.blip.state_dict())
    model.denoised_mlp.load_state_dict(model.mlp.state_dict())
    model.mlp.eval()
    
    model.freeze_original_models()
    print("Checkpoint loaded and original models frozen")
    model.freeze_denoised_layers(fix_rate=0.7)

    return model

def validate_model_modes(model):
    """Verify that original MLP is in eval mode while denoised_mlp respects training mode"""
    if hasattr(model, 'module'):
        model = model.module
    
    if model.mlp.training:
        print("Warning: Original MLP is in training mode when it should be in eval mode")
        
    if model.denoised_mlp.training != model.training:
        print(f"Warning: denoised_mlp training mode ({model.denoised_mlp.training}) "
              f"doesn't match model training mode ({model.training})")
    
    for name, module in model.denoised_mlp.named_modules():
        if isinstance(module, nn.Dropout):
            if module.training != model.training:
                print(f"Warning: Dropout layer {name} in denoised_mlp has incorrect training mode")

def decompose_and_batch_samples_list(samples_list, batch_size):
    """Decompose samples list, shuffle, and create batches of specified size."""
    all_decomposed = []
    for sample in samples_list:
        batch_size_orig = len(sample["prompt"])
        num_steps = sample["timesteps"].shape[1]
        
        for b in range(batch_size_orig):
            for t in range(num_steps):
                single_sample = {
                    "prompt": sample["prompt"][b],
                    "prompt_embeds": sample["prompt_embeds"][b],
                    "latents": sample["latents"][b,t],
                    "denoised_latents": sample["denoised_latents"][b,t],
                    "timesteps": sample["timesteps"][b,t],
                    "rewards": sample["rewards"][b]
                }
                all_decomposed.append(single_sample)
    
    import random
    random.shuffle(all_decomposed)
    
    batched_samples = []
    total_samples = len(all_decomposed)
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch = all_decomposed[start_idx:end_idx]
        
        batched_sample = {
            "prompt": [item["prompt"] for item in batch],
            "prompt_embeds": torch.stack([item["prompt_embeds"] for item in batch]),
            "latents": torch.stack([item["latents"] for item in batch]),
            "denoised_latents": torch.stack([item["denoised_latents"] for item in batch]),
            "timesteps": torch.stack([item["timesteps"] for item in batch]),
            "rewards": torch.tensor([item["rewards"] for item in batch])
        }
        batched_samples.append(batched_sample)
    
    return batched_samples

def train_value_batch(value_function, samples_batched, pipeline, accelerator, config):
    """Train value function on a single batch."""
    if accelerator.is_main_process:
        validate_model_modes(value_function)

    batch_data = samples_batched
    latents = batch_data["latents"].to(accelerator.device)
    denoised_latents = batch_data["denoised_latents"].to(accelerator.device)
    batch_final_reward = batch_data["rewards"].to(accelerator.device)
    prompts = batch_data["prompt"]
    timesteps = batch_data["timesteps"].to(accelerator.device)

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    with torch.no_grad():
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = resize_transform(images)
    
        denoised_images = pipeline.vae.decode(denoised_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        denoised_images = (denoised_images + 1) / 2
        denoised_images = denoised_images.clamp(0, 1)
        denoised_images = resize_transform(denoised_images)
    
    pred_value = value_function(images, denoised_images, prompts, timesteps) + 0.0 * sum([torch.sum(param) for param in value_function.parameters()])
    value_loss = F.mse_loss(pred_value.float(), batch_final_reward.float())
    accelerator.backward(value_loss)
    return value_loss

def init_value_function(config, accelerator):
    """Initialize value function, optimizer and scheduler"""
    if config.pretrained_VN_path:
        value_function = load_VN(config.pretrained_VN_path, med_config=config.value_network.med_config, device=accelerator.device)
        print("loaded pretrained value function")
    else:
        value_function = load("ImageReward-v1.0", med_config=config.value_network.med_config, device=accelerator.device)
    
    if hasattr(value_function, "gradient_checkpointing_enable"):
        value_function.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for value network")
    
    warmup_ratio = config.pretrain_VN.warmup_ratio
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, value_function.parameters()),
        lr=config.pretrain_VN.learning_rate,
        betas=(config.pretrain_VN.adam_beta1, config.pretrain_VN.adam_beta2),
        weight_decay=config.pretrain_VN.adam_weight_decay,
        eps=config.pretrain_VN.adam_epsilon,
    )

    # Define a constant function for learning rate
    constant_lr_lambda = lambda epoch: 1.0  # Always return 1.0

    # Use LambdaLR to apply constant learning rate
    scheduler = LambdaLR(optimizer, lr_lambda=constant_lr_lambda)
    value_function, optimizer = accelerator.prepare(value_function, optimizer)

    if isinstance(value_function, torch.nn.parallel.DistributedDataParallel):
        value_function.find_unused_parameters = True
        
    return value_function, optimizer, scheduler

def inference_value_batch(value_function, sample_dict, pipeline, accelerator, sub_batch_size=50):
    """Train value function on batches to avoid memory issues.
    
    Args:
        value_function: The value function model
        sample_dict: Dictionary containing batch data
        pipeline: Pipeline for processing
        accelerator: Accelerator for device management
        sub_batch_size: Number of samples to process in each sub-batch
    
    Returns:
        Tensor of shape (batch_size, num_samples, output_dim)
    """
    if accelerator.is_main_process:
        validate_model_modes(value_function)

    batch_data = sample_dict
    latents = batch_data["latents"].to(accelerator.device)
    denoised_latents = batch_data["denoised_latents"].to(accelerator.device)
    prompts = batch_data["prompt"]
    timesteps = batch_data["timesteps"].to(accelerator.device)

    batch_size, num_samples = latents.shape[0], latents.shape[1]
    total_samples = batch_size * num_samples
    
    # Reshape inputs
    latents = latents.reshape(-1, 4, 64, 64)
    denoised_latents = denoised_latents.reshape(-1, 4, 64, 64)
    timesteps = timesteps.reshape(-1)
    prompts = [p for p in prompts for _ in range(num_samples)]

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])

    # Process in sub-batches
    all_values = []
    for i in range(0, total_samples, sub_batch_size):
        end_idx = min(i + sub_batch_size, total_samples)
        
        with torch.no_grad():
            # Process current sub-batch
            sub_latents = latents[i:end_idx]
            sub_denoised = denoised_latents[i:end_idx]
            sub_prompts = prompts[i:end_idx]
            sub_timesteps = timesteps[i:end_idx]

            # Decode images
            images = pipeline.vae.decode(sub_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            images = (images + 1) / 2
            images = images.clamp(0, 1)
            images = resize_transform(images)

            denoised_images = pipeline.vae.decode(sub_denoised / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            denoised_images = (denoised_images + 1) / 2
            denoised_images = denoised_images.clamp(0, 1)
            denoised_images = resize_transform(denoised_images)

            # Get value function output for sub-batch
            sub_values = value_function(images, denoised_images, sub_prompts, sub_timesteps)
            all_values.append(sub_values)

            # Free memory
            del images, denoised_images, sub_values
            torch.cuda.empty_cache()

    # Concatenate all sub-batch results
    values = torch.cat(all_values, dim=0)
    
    # Reshape to original batch dimensions
    values = values.reshape(batch_size, num_samples)
    
    return values
    