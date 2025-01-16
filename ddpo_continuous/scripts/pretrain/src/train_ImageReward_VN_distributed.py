import os
import datetime
import time
from collections import defaultdict
import torch
import torch.nn.functional as F
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob, pipeline_with_denoised_latents_logprob
import wandb
from functools import partial
import tqdm
from PIL import Image
from concurrent import futures
from ImageReward_VN import ImageRewardValue
from torchvision import transforms
from huggingface_hub import hf_hub_download
from typing import Any, Union, List
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
}

def prepare_latent_batch(samples_batched):
    """Prepare latent batch for training with proper type checking."""
    latents_list = []
    denoised_latents_list = []
    rewards_list = []
    prompts_list = []
    timesteps_list = []

    # Handle both single samples and batched samples
    if not isinstance(samples_batched, (list, tuple)):
        samples_batched = [samples_batched]

    for sample in samples_batched:
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dictionary for sample, got {type(sample)}")
            
        # Verify all required keys are present
        required_keys = ["latents", "denoised_latents", "rewards", "prompt", "timesteps"]
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise KeyError(f"Missing required keys in sample: {missing_keys}")

        latents_list.append(sample["latents"].contiguous())
        denoised_latents_list.append(sample["denoised_latents"].contiguous())
        rewards_list.append(sample["rewards"].contiguous())
        
        if isinstance(sample["prompt"], (list, tuple)):
            prompts_list.extend(sample["prompt"])
        else:
            prompts_list.append(sample["prompt"])
            
        timesteps_list.append(sample["timesteps"].contiguous())

    # Stack tensors along batch dimension
    latents = torch.cat(latents_list, dim=0)
    denoised_latents = torch.cat(denoised_latents_list, dim=0)
    rewards = torch.cat(rewards_list, dim=0)
    timesteps = torch.cat(timesteps_list, dim=0)

    return {
        "latents": latents,
        "denoised_latents": denoised_latents,
        "rewards": rewards,
        "prompts": prompts_list,
        "timesteps": timesteps,
    }

def create_dataloader(samples, batch_size, accelerator):
    """Create distributed dataloader with proper error checking."""
    if not samples:
        raise ValueError("Empty samples list provided to create_dataloader")
    
    # Verify sample structure before creating dataset
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dictionary for sample, got {type(sample)}")
        required_keys = ["latents", "denoised_latents", "rewards", "prompt", "timesteps"]
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise KeyError(f"Missing required keys in sample: {missing_keys}")
    
    dataset = SamplesDataset(samples)
    
    # Calculate local batch size
    local_batch_size = max(1, batch_size // accelerator.num_processes)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=lambda x: x  # Use identity collate function to prevent automatic batching
    )
    
    if len(dataloader) == 0:
        raise ValueError(
            f"Empty dataloader created. Samples: {len(samples)}, "
            f"Batch size: {batch_size}, Local batch size: {local_batch_size}"
        )
    
    return dataloader

class SamplesDataset(Dataset):
    """Dataset class with proper data validation."""
    def __init__(self, samples):
        if not isinstance(samples, (list, tuple)):
            raise TypeError(f"Expected list or tuple of samples, got {type(samples)}")
        self.samples = samples
        
        # Validate all samples have the correct format
        for sample in samples:
            if not isinstance(sample, dict):
                raise TypeError(f"Expected dictionary for sample, got {type(sample)}")
            required_keys = ["latents", "denoised_latents", "rewards", "prompt", "timesteps"]
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                raise KeyError(f"Missing required keys in sample: {missing_keys}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def available_models() -> List[str]:
    return list(_MODELS.keys())

def ImageReward_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target

def load(name: str = "ImageReward-v1.0", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
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
    model.freeze_original_models()
    print("Checkpoint loaded and original models frozen")
    return model

def ensure_contiguous(sample):
    """Ensure all tensors in the sample dictionary are contiguous."""
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.contiguous()
    return sample

def compute_rewards(images, prompts, prompt_metadata, reward_fn):
    """Compute rewards using the reward function."""
    # Convert tensor images to PIL images
    pil_images = []
    for image in images:
        image_array = (image.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype("uint8")
        pil_images.append(Image.fromarray(image_array))
    
    # Convert PIL images to tensor format expected by reward_fn
    image_tensors = torch.stack([
        transforms.ToTensor()(img) for img in pil_images
    ]).cuda()
    
    rewards = reward_fn(image_tensors, prompts, prompt_metadata)
    return rewards

def gather_samples(samples, accelerator):
    """Gather samples from all processes with proper tensor handling."""
    gathered_samples = []
    
    for sample in samples:
        # Ensure all tensors are contiguous
        gathered_sample = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                gathered_value = accelerator.gather(value.contiguous())
                gathered_sample[key] = gathered_value
            elif isinstance(value, (list, tuple)):
                # Handle non-tensor data like prompts
                gathered_value = accelerator.gather_for_metrics(value)
                gathered_sample[key] = gathered_value
            else:
                gathered_sample[key] = value
        gathered_samples.append(gathered_sample)
    
    return gathered_samples

def train_value_batch(value_function, samples_batched, pipeline, accelerator, config):
    """Train value function on a batch of samples with improved error handling."""
    if not samples_batched:
        raise ValueError("Empty batch provided to train_value_batch")
    
    batch_data = prepare_latent_batch(samples_batched)
    
    # Validate batch data
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor) and value.numel() == 0:
            raise ValueError(f"Empty tensor found in batch_data for key: {key}")
    
    latents = batch_data["latents"].to(accelerator.device)
    latents = latents.reshape(-1, 4, 64, 64)
    
    denoised_latents = batch_data["denoised_latents"].to(accelerator.device)
    denoised_latents = denoised_latents.reshape(-1, 4, 64, 64)
    
    batch_final_reward = batch_data["rewards"].to(accelerator.device).repeat(config.sample.num_steps)
    prompts = [item for item in batch_data["prompts"] for _ in range(config.sample.num_steps)]
    timesteps = batch_data["timesteps"].to(accelerator.device).reshape(-1)
    
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
    
    with accelerator.accumulate(value_function):
        pred_value = value_function(images, denoised_images, prompts, timesteps).squeeze(-1)
        value_loss = F.mse_loss(pred_value.float(), batch_final_reward.float())
        accelerator.backward(value_loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(value_function.parameters(), config.train.max_grad_norm)
    
    return value_loss.item()

def main(_):
    config = FLAGS.config
    if not hasattr(config, 'v_step') or config.v_step <= 0:
        raise ValueError("config.v_step must be a positive integer")
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = f"{config.run_name}_{unique_id}" if config.run_name else unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.sample.num_steps
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="image-reward-value",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}}
        )
    
    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(accelerator.device)
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    local_sample_batch_size = config.sample.batch_size // accelerator.num_processes
    local_train_batch_size = config.train.batch_size // accelerator.num_processes
    
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(local_sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(local_train_batch_size, 1, 1)

    value_function = load("ImageReward-v1.0", med_config=config.value_network.med_config, device=accelerator.device)
    value_function.freeze_denoised_layers(fix_rate=0.7)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, value_function.parameters()),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    value_function, optimizer = accelerator.prepare(value_function, optimizer)
    executor = futures.ThreadPoolExecutor(max_workers=2)

    global_step = 0
    for epoch in range(config.num_epochs):
        pipeline.unet.eval()
        samples = []

        for batch_idx in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling"):
            local_prompts, local_prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(local_sample_batch_size)]
            )

            prompt_ids = pipeline.tokenizer(
                local_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)

            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            
            with torch.no_grad(), accelerator.autocast():
                current_neg_prompt_embeds = sample_neg_prompt_embeds[:prompt_embeds.size(0)]
                
                images, _, latents, denoised_latents, log_probs = pipeline_with_denoised_latents_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=current_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
            
            latents = torch.stack(latents, dim=1).contiguous()
            denoised_latents = torch.stack(denoised_latents, dim=1).contiguous()
            log_probs = torch.stack(log_probs, dim=1).contiguous()
            
            timesteps = pipeline.scheduler.timesteps.repeat(local_sample_batch_size, 1).contiguous()

            rewards = executor.submit(
                compute_rewards, 
                images,
                local_prompts,
                local_prompt_metadata,
                reward_fn
            )
            
            samples.append(ensure_contiguous({
                "prompt_ids": prompt_ids,
                "prompt": local_prompts,
                "prompt_embeds": prompt_embeds,
                "latents": latents[:,:-1],
                "denoised_latents": denoised_latents,
                "timesteps": timesteps,
                "rewards": rewards
            }))

        # Process rewards
        for sample in tqdm(samples, desc="Waiting for rewards", disable=not accelerator.is_local_main_process):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device).contiguous()

        # Gather samples across processes
        gathered_samples = gather_samples(samples, accelerator)
        if accelerator.is_main_process:
            samples = gathered_samples

        # Training phase
        value_function.train()
        train_dataloader = create_dataloader(samples, config.train.batch_size, accelerator)

        for inner_epoch in range(config.train.num_inner_epochs):
            total_value_loss = 0
            num_batches = 0
            
            train_dataloader.sampler.set_epoch(inner_epoch)
            
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Inner epoch {inner_epoch}",
                disable=not accelerator.is_local_main_process,
            )

            for batch_samples in train_dataloader:
                optimizer.zero_grad()
                
                for v_step in range(config.v_step):
                    if v_step < config.v_step - 1:
                        with accelerator.no_sync(value_function):
                            value_loss = train_value_batch(
                                value_function,
                                batch_samples,
                                pipeline,
                                accelerator,
                                config
                            )
                    else:
                        value_loss = train_value_batch(
                            value_function,
                            batch_samples,
                            pipeline,
                            accelerator,
                            config
                        )
                    total_value_loss += value_loss
                    num_batches += 1

                optimizer.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)

            if num_batches > 0:
                avg_loss = total_value_loss / (num_batches * config.v_step)
                avg_loss = accelerator.gather(torch.tensor(avg_loss, device=accelerator.device)).mean().item()
                
                if accelerator.is_main_process:
                    logger.info(f"Epoch={epoch}, Inner={inner_epoch}, Loss={avg_loss:.4f}")
                    accelerator.log(
                        {
                            "train/value_loss": avg_loss,
                            "train/epoch": epoch,
                            "train/inner_epoch": inner_epoch,
                        },
                        step=global_step,
                    )
            progress_bar.close()
            global_step += 1

        # Save checkpoints
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            checkpoint_dir = os.path.join(
                accelerator.project_configuration.project_dir,
                f"checkpoint_{epoch}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            accelerator.save(
                value_function.state_dict(),
                os.path.join(checkpoint_dir, "value_network.pt")
            )
            
            accelerator.save(
                optimizer.state_dict(),
                os.path.join(checkpoint_dir, "optimizer.pt")
            )
            
            accelerator.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": config.to_dict(),
                },
                os.path.join(checkpoint_dir, "training_state.pt")
            )

    executor.shutdown()
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)