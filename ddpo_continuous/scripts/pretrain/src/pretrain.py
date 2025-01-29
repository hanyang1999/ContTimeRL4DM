import os
import datetime
import time
from concurrent import futures
import torch
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm.auto import tqdm
from functools import partial
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_denoised_latents_logprob
from ImageReward_VN_train import (
    init_value_function,
    train_value_batch,
    decompose_and_batch_samples_list
)

logger = get_logger(__name__)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
tqdm = partial(tqdm, dynamic_ncols=True)

def main(_):
    # Initialize config and logging
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

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
        gradient_accumulation_steps=config.pretrain_VN.gradient_accumulation_steps * config.sample.num_steps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="Value-function Pretraining",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name, "entity": "fantastic_team"}}
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

    # Set up inference dtype
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
    # Initialize reward components
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    # Generate negative prompt embeddings
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

    # Initialize value network components
    value_function, optimizer, scheduler = init_value_function(config, accelerator)
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Training loop setup
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.pretrain_VN.batch_size * accelerator.num_processes * config.pretrain_VN.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Num Processes = {accelerator.num_processes}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.pretrain_VN.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.pretrain_VN.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.pretrain_VN.num_inner_epochs}")

    assert (config.sample.batch_size * config.sample.num_steps) >= config.pretrain_VN.batch_size
    assert (config.sample.batch_size * config.sample.num_steps) % config.pretrain_VN.batch_size == 0
    assert (samples_per_epoch * config.sample.num_steps) % total_train_batch_size == 0

    # Training loop
    global_step = 0
    for epoch in tqdm(range(config.num_epochs), desc="Epoch"):
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.unet.eval()
        samples = []

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
                "prompt": prompts,
                "prompt_embeds": prompt_embeds,
                "latents": latents[:,:-1],
                "denoised_latents": denoised_latents,
                "timesteps": timesteps,
                "rewards": rewards
            })

        # Wait for rewards computation
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # Training phase
        value_function.train()

        for inner_epoch in range(config.pretrain_VN.num_inner_epochs):
            total_value_loss = 0
            counts = 0
            optimizer.zero_grad()

            # Calculate total number of iterations for the progress bar
            total_iters = sum(len(decompose_and_batch_samples_list(samples[i:i+1], config.pretrain_VN.batch_size)) * config.v_step 
                            for i in range(len(samples)))
            
            # Create progress bar
            pbar = tqdm(total=total_iters, 
                       desc=f"Inner Epoch {inner_epoch+1}/{config.pretrain_VN.num_inner_epochs}",
                       leave=False)
            
            # Train on decomposed samples
            for idx in range(len(samples)):
                batch_vf_batch_samples = decompose_and_batch_samples_list(
                    samples[idx:idx+1], 
                    config.pretrain_VN.batch_size
                )
                
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
                            accelerator.clip_grad_norm_(
                                value_function.parameters(), 
                                config.pretrain_VN.max_grad_norm
                            )
                            
                        optimizer.step()
                        optimizer.zero_grad()

                        current_loss = value_loss.detach().item()
                        total_value_loss += current_loss
                        counts += 1
                        
                        pbar.set_postfix({
                            'loss': f'{current_loss:.4f}', 
                            'avg_loss': f'{(total_value_loss/counts):.4f}'
                        })
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
            os.makedirs(accelerator.project_configuration.project_dir, exist_ok=True)
            accelerator.save(
                value_function.state_dict(),
                os.path.join(
                    accelerator.project_configuration.project_dir, 
                    f"value_network_epoch_{epoch}.pt"
                )
            )

    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)