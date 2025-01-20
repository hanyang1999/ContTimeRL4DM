import os
import datetime
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
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
from transformers import get_scheduler
from typing import Any, Union, List
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"



_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
}


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


logger = get_logger(__name__)
FLAGS = flags.FLAGS
# config_flags.DEFINE_CONFIG_FILE("config", "config/base.py", "Training configuration.")
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

def bp():
    import pdb; pdb.set_trace()


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
    """Load a ImageReward model with dual BLIP architecture

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model with frozen original BLIP/MLP and learnable denoised BLIP/MLP
    """
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('Loading checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Load med_config
    if med_config is None:
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", 
                                        download_root or os.path.expanduser("~/.cache/ImageReward"))
    
    # Initialize model with dual BLIP architecture
    model = ImageRewardValue(med_config=med_config).to(device)
    
    # Load state dict into original (frozen) BLIP and MLP
    # The strict=False allows loading the original weights without denoised BLIP parameters
    msg = model.load_state_dict(state_dict, strict=False)

    # Initialize denoised BLIP with same weights as original BLIP
    model.denoised_blip.load_state_dict(model.blip.state_dict())
    model.denoised_mlp.load_state_dict(model.mlp.state_dict())

    model.mlp.eval()
    
    # Freeze original models
    model.freeze_original_models()
    print("Checkpoint loaded and original models frozen")

    model.freeze_denoised_layers(fix_rate=0.7)  # Freeze 70% of layers in denoised BLIP

    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    return model

def validate_model_modes(model):
    """Verify that original MLP is in eval mode while denoised_mlp respects training mode"""
    
    # Get the actual model from DDP wrapper if necessary
    if hasattr(model, 'module'):
        model = model.module
    
    # Check original MLP is always in eval mode
    if model.mlp.training:
        print("Warning: Original MLP is in training mode when it should be in eval mode")
        
    # Check denoised_mlp follows model's training mode
    if model.denoised_mlp.training != model.training:
        print(f"Warning: denoised_mlp training mode ({model.denoised_mlp.training}) "
              f"doesn't match model training mode ({model.training})")
    
    # Check dropout layers specifically
    for name, module in model.denoised_mlp.named_modules():
        if isinstance(module, nn.Dropout):
            if module.training != model.training:
                print(f"Warning: Dropout layer {name} in denoised_mlp has incorrect training mode")
                

def compute_rewards(images, prompts, prompt_metadata, reward_fn):
    """Compute rewards using the reward function."""
    pil_images = []
    for image in images:
        image_array = (image.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype("uint8")
        pil_images.append(Image.fromarray(image_array))

    rewards = reward_fn(pil_images, prompts, prompt_metadata)
    return rewards


def decompose_and_batch_samples_list(samples_list, batch_size):
    """
    Decompose samples list, shuffle, and create batches of specified size.
    Returns a list of batched samples, each with batch_size items 
    (except possibly the last batch which might be smaller).
    """
    # First decompose all samples
    all_decomposed = []
    for sample in samples_list:
        batch_size_orig = len(sample["prompt"])
        num_steps = sample["timesteps"].shape[1]
        
        for b in range(batch_size_orig):
            for t in range(num_steps):
                single_sample = {
                    # "prompt_ids": sample["prompt_ids"][b],
                    "prompt": sample["prompt"][b],
                    "prompt_embeds": sample["prompt_embeds"][b],
                    "latents": sample["latents"][b,t],
                    "denoised_latents": sample["denoised_latents"][b,t],
                    "timesteps": sample["timesteps"][b,t],
                    "rewards": sample["rewards"][b]
                }
                all_decomposed.append(single_sample)
    
    # Randomly shuffle the decomposed samples
    import random
    random.shuffle(all_decomposed)
    
    # Create batches
    batched_samples = []
    total_samples = len(all_decomposed)
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch = all_decomposed[start_idx:end_idx]
        
        # Create a batched sample dictionary
        batched_sample = {
            # "prompt_ids": [item["prompt_ids"] for item in batch],
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

    # Check model modes
    if accelerator.is_main_process:
        validate_model_modes(value_function)

    """Train value function on a single batch."""
    batch_data = samples_batched

    latents = batch_data["latents"].to(accelerator.device)

    denoised_latents = batch_data["denoised_latents"].to(accelerator.device)

    batch_final_reward = batch_data["rewards"].to(accelerator.device)

    prompts = batch_data["prompt"]

    timesteps = batch_data["timesteps"].to(accelerator.device)\

    # Add image resizing transform
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Decode latents to images using the VAE
    with torch.no_grad():
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        # Convert from [-1,1] to [0,1] range for BLIP
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = resize_transform(images)
    
    with torch.no_grad():
        denoised_images = pipeline.vae.decode(denoised_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        # Convert from [-1,1] to [0,1] range for BLIP
        denoised_images = (denoised_images + 1) / 2
        denoised_images = denoised_images.clamp(0, 1)
        denoised_images = resize_transform(denoised_images)
    
    # Forward pass using decoded images
    pred_value = value_function(images, denoised_images, prompts, timesteps) + 0.0 * sum([torch.sum(param) for param in value_function.parameters()])
    value_loss = F.mse_loss(pred_value.float(), batch_final_reward.float())
    accelerator.backward(value_loss)
    return value_loss

def main(_):
    # Initialize config and logging
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = f"{config.run_name}_{unique_id}" if config.run_name else unique_id

    # Setup accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.sample.num_steps,
        #partial_pipeline_kwargs={"find_unused_parameters": True},
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="image-reward-value",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}}
        )
    
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # Initialize pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(accelerator.device)
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
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

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
    # Initialize reward components
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # # initialize stat tracker
    # if config.per_prompt_stat_tracking:
    #     stat_tracker = PerPromptStatTracker(
    #         config.per_prompt_stat_tracking.buffer_size,
    #         config.per_prompt_stat_tracking.min_count,
    #     )

    # Initialize value network and freeze layers
    # Usage in training script
    value_function = load("ImageReward-v1.0", med_config=config.value_network.med_config, device=accelerator.device)
    

    if hasattr(value_function, "gradient_checkpointing_enable"):
        value_function.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for value network")
    
    warmup_ratio = 0.1         # 10% of total steps for warmup
    num_warmup_steps = int(config.num_epochs * warmup_ratio)
    
    # Initialize optimizer for unfrozen parameters only
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, value_function.parameters()),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    scheduler = get_scheduler(
    name="cosine",  # Options: 'linear', 'cosine', 'polynomial', etc.
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=config.num_epochs,
)

    # Prepare with accelerator
    value_function, optimizer = accelerator.prepare(value_function, optimizer)

    if isinstance(value_function, torch.nn.parallel.DistributedDataParallel):
        value_function.find_unused_parameters = True  # Enable unused parameter detection
    
    executor = futures.ThreadPoolExecutor(max_workers=2)

    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Num Processes = {accelerator.num_processes}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    # Training loop
    global_step = 0
    for epoch in tqdm(range(config.num_epochs),desc="Epoch"):
        pipeline.unet.eval()
        samples = []
        prompts = []

        # Sampling phase
        for batch_idx in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling"):
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )

            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)

            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            
            with torch.no_grad(), accelerator.autocast():
                images, _, latents, denoised_latents, log_probs = pipeline_with_denoised_latents_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
            
            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            denoised_latents = torch.stack(denoised_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)

            rewards = executor.submit(
                reward_fn, 
                images, 
                prompts, 
                prompt_metadata,
            )
                
            time.sleep(0)

            samples.append({
                # "prompt_ids": prompt_ids,
                "prompt": prompts, # list of strings with length = sample.batchsize
                "prompt_embeds": prompt_embeds, 
                "latents": latents[:,:-1],  # sample.batchsize * num_steps * 4 * 64 * 64
                "denoised_latents": denoised_latents,  # sample.batchsize * num_steps * 4 * 64 * 64
                "timesteps": timesteps, # sample.batchsize * num_steps
                "rewards": rewards # sample.batchsize
            })

        # Wait for rewards computation
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata  = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        # samples["prompts"] = prompts

        # Training phase
        value_function.train()

        for inner_epoch in range(config.train.num_inner_epochs):
            total_value_loss = 0
            counts = 0
            optimizer.zero_grad()

            # Calculate total number of iterations for the progress bar
            total_iters = sum(len(decompose_and_batch_samples_list(samples[i:i+1], config.train.value_function_batch_size)) * config.v_step 
                            for i in range(len(samples)))
            
            # Create progress bar
            pbar = tqdm(total=total_iters, 
                        desc=f"Inner Epoch {inner_epoch+1}/{config.train.num_inner_epochs}",
                        leave=False)  # leave=False means the bar will be cleared after completion
            
            # Decompose samples into single samples
            for idx in range(len(samples)):
                batch_vf_batch_samples = decompose_and_batch_samples_list(samples[idx:idx+1], config.train.value_function_batch_size)
                for vf_batch_samples in batch_vf_batch_samples:
                    for _ in range(config.v_step):
                        value_loss = train_value_batch(
                                value_function, 
                                vf_batch_samples, 
                                pipeline,
                                accelerator, 
                                config
                            )
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(value_function.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                        current_loss = value_loss.detach().item()
                        total_value_loss += current_loss
                        counts += 1
                        
                        # Update progress bar with current loss
                        pbar.set_postfix({'loss': f'{current_loss:.4f}', 
                                        'avg_loss': f'{(total_value_loss/counts):.4f}'})
                        pbar.update(1)
                        
                        del value_loss
            scheduler.step()
            pbar.close()

            # Logging
            avg_loss = total_value_loss / counts
            logger.info(f"Epoch={epoch}, Inner={inner_epoch}, Loss={avg_loss:.4f}")
            if accelerator.is_main_process:
                accelerator.log({"value_loss": avg_loss}, step=global_step)
            global_step += 1

        # Save checkpoint
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            # accelerator.save_state()
            os.makedirs(accelerator.project_configuration.project_dir, exist_ok=True)
            accelerator.save(
                value_function.state_dict(),
                os.path.join(accelerator.project_configuration.project_dir, f"value_network_epoch_{epoch}.pt")
            )

    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)