import os
import datetime
import time
from concurrent import futures
from collections import defaultdict
import contextlib
import tempfile
import copy
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
import wandb
import tqdm
from PIL import Image
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from torchvision import transforms
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_denoised_latents_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ImageReward_VN_train import init_value_function, train_value_batch

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)

def _train_value_func(value_function, samples_VN, pipeline, accelerator, config):
    """Train value function using ImageReward with dedicated VN samples."""
    latents = samples_VN["latents"]
    next_latents = samples_VN["next_latents"]
    rewards = samples_VN["rewards"]
    timesteps = samples_VN["timesteps"]
    text_embeddings = samples_VN["text_embeddings"]
    
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])

    # Decode latents
    with torch.no_grad():
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        images = resize_transform((images + 1) / 2).clamp(0, 1)
        
        next_images = pipeline.vae.decode(next_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        next_images = resize_transform((next_images + 1) / 2).clamp(0, 1)
    
    # Forward pass
    pred_value = value_function(images, next_images, text_embeddings, timesteps)
    value_loss = F.mse_loss(pred_value.float(), rewards.float())
    accelerator.backward(value_loss/config.v_step)
    
    return value_loss.item() / config.v_step

def batch_samples(samples_list, batch_size):
    """Convert list of samples into batched format."""
    if not samples_list:
        return {}
    
    batched = {}
    for key in samples_list[0].keys():
        # Stack all samples for this key
        stacked = torch.stack([s[key] for s in samples_list]) if torch.is_tensor(samples_list[0][key]) else [s[key] for s in samples_list]
        if torch.is_tensor(stacked):
            # Reshape into batches
            batched[key] = stacked.reshape(-1, batch_size, *stacked.shape[2:])
    return batched

def main(_):
    # Initialize config and logging
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = f"{config.run_name}_{unique_id}" if config.run_name else unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # Initialize accelerator and devices
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name, "entity": "contiRL4diffusion"}}
        )
    logger.info(f"\n{config}")


    # Set random seed
    set_seed(config.seed, device_specific=True)

    # Initialize models
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(accelerator.device)
    
    pipeline_original = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision
    )

    pipeline_original.eval()
    
    # Freeze parameters
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Initialize value function
    value_function, value_optimizer, scheduler = init_value_function(config, accelerator)

    # Set up mixed precision
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    # Initialize LoRA if needed
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)
        
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)
        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # Initialize optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # Prepare everything with accelerator
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Initialize other components
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    executor = futures.ThreadPoolExecutor(max_workers=2)

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


    # Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        value_function.eval()
        
        samples_VN = []
        samples_PN = []
        prompts = []
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # Generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            
            # Get embeddings for both networks
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            
            # Sample images
            with autocast():
                images, _, latents, denoised_latents, log_probs = pipeline_with_denoised_latents_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
            
            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)
            
            # Compute rewards
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            time.sleep(0)

            samples_PN.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

            samples_VN.append({
                "prompt": prompts,
                "latents": latents[:,:-1],
                "denoised_latents": denoised_latents,
                "timesteps": timesteps,
                "rewards": rewards
            })

        # Process rewards
        for sample in tqdm(
            samples_VN,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        for sample in tqdm(
            samples_PN,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
        
        samples_PN = {k: torch.cat([s[k] for s in samples_PN]) for k in samples_PN[0].keys()}

        # Log sample images
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            
            accelerator.log({
                "images": [
                    wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                    for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                ],
            }, step=global_step)

        # gather rewards across processes
        rewards = accelerator.gather(samples_PN["rewards"]).cpu().numpy()

        # Log rewards and images
        accelerator.log({
            "reward": rewards,
            "epoch": epoch,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std()
        }, step=global_step)

        # Compute advantages
        # if config.use_regularization:
        #     rewards_for_advantage = torch.cat([s["rewards_regularized"] for s in samples_VN_list])
        # else:
        #     rewards_for_advantage = all_rewards

        # advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Assign advantages to policy network samples
        # advantages = advantages.reshape(accelerator.num_processes, -1)[accelerator.process_index].to(accelerator.device)
        
        # samples_PN["advantages"] = (
        #     torch.as_tensor(advantages)
        #     .reshape(accelerator.num_processes, -1)[accelerator.process_index]
        #     .to(accelerator.device)
        # )

        samples_PN["advantages"] = (
            torch.as_tensor(rewards)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        #################### VALUE FUNCTION TRAINING ####################
        if config.v_flag == 1:
            pipeline.unet.eval()
            value_function.train()
            
            # Batch samples for value network
            batched_VN_samples = batch_samples(samples_VN, config.train.batch_size)
            
            tot_val_loss = 0
            value_optimizer.zero_grad()
            
            for v_step in range(config.v_step):
                if v_step < config.v_step - 1:
                    with accelerator.no_sync(value_function):
                        tot_val_loss += _train_value_func(
                            value_function,
                            batched_VN_samples,
                            pipeline,
                            accelerator,
                            config
                        )
                else:
                    tot_val_loss += _train_value_func(
                        value_function,
                        batched_VN_samples,
                        pipeline,
                        accelerator,
                        config
                    )
            
            value_optimizer.step()
            value_optimizer.zero_grad()
            
            if accelerator.is_main_process:
                accelerator.log({"value_loss": tot_val_loss}, step=global_step)
            
            del tot_val_loss
            torch.cuda.empty_cache()

        #################### POLICY TRAINING ####################
        value_function.eval()
        pipeline.unet.train()

        for inner_epoch in range(config.train.num_inner_epochs):
            # Prepare batched samples for policy training
            batched_PN_samples = batch_samples(samples_PN, config.train.batch_size)
            info = defaultdict(list)

            for i, sample in enumerate(batched_PN_samples):
                if config.train.cfg:
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
                for j in tqdm(range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            # Forward pass
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample

                            # Compute log probabilities
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                            # Get values from value function
                            scaled_timestep = (sample["timesteps"][:, j] - 1) * config.sample.num_steps / 1000
                            
                            with torch.no_grad():
                                value_current = value_function(
                                    sample["latents"][:, j],
                                    sample["next_latents"][:, j],
                                    sample["text_embeddings"],
                                    scaled_timestep.int()
                                )
                                next_timestep = torch.max(scaled_timestep - 1, torch.zeros_like(scaled_timestep))
                                value_next = value_function(
                                    sample["next_latents"][:, j],
                                    sample["next_latents"][:, j],
                                    sample["text_embeddings"],
                                    next_timestep.int()
                                )

                            # Compute advantages and TD error
                            # if config.use_regularization:
                            #     TD_error = sample["regularization_terms"][:, j] + value_next - value_current
                            #     advantages = TD_error
                            # else:
                            advantages = sample["advantages"]
                            advantages = torch.clamp(
                                advantages, -config.train.adv_clip_max, config.train.adv_clip_max
                            )

                            # PPO update
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                            )
                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            # Store metrics
                            info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                            info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                            info["loss"].append(loss)

                            # Backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                    # Log if we did an optimization step
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

        # Save checkpoints
        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()
            
            # Save value function separately
            accelerator.save(
                value_function.state_dict(),
                os.path.join(
                    accelerator.project_configuration.project_dir,
                    f"value_function_epoch_{epoch}.pt"
                )
            )

    # End training
    accelerator.end_training()
    executor.shutdown()

if __name__ == "__main__":
    app.run(main)