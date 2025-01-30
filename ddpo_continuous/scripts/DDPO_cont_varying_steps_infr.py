from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob, pipeline_with_logprob_regularizedReward, pipeline_with_denoised_latents_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from util import get_decayed_value


# import value function networks
from src.ImageReward_VN_train import (
    init_value_function,
    train_value_batch,
    decompose_and_batch_samples_list,
    inference_value_batch,
    inference_advantage_batch
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def bp():
    import pdb; pdb.set_trace()

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
    
    if config.pretrained_VN_path:
        config.pretrained_VN_path = os.path.normpath(os.path.expanduser(config.pretrained_VN_path))

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # accumulate gradients across timesteps as in DDPO
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )


    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="RL training for DM", config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name, "entity": "fantastic_team"}}
        )
    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, 
        revision=config.pretrained.revision,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(accelerator.device)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
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
        # pipeline_original.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
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

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet
    
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 2
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 2
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # omit the 8_bit_adam option as in DDPO
    optimizer_cls = torch.optim.AdamW

    unet_optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
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


    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, unet_optimizer = accelerator.prepare(unet, unet_optimizer)

    # Initialize value network components
    value_function, value_optimizer, value_opt_scheduler = init_value_function(config, accelerator)
    
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
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

    assert (config.sample.batch_size * config.sample.num_steps) >= config.pretrain_VN.batch_size
    assert (config.sample.batch_size * config.sample.num_steps) % config.pretrain_VN.batch_size == 0
    assert (samples_per_epoch * config.sample.num_steps) % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0

    var_inf_steps = [25, 100]

    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        samples_rewards = {}
        var_inf_images = {}
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            for var_inf_iter, num_steps in enumerate(var_inf_steps):
                with torch.no_grad(),autocast():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=get_decayed_value(epoch, config),  #config.sample.eta,
                        output_type="pt",
                    )
                
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
                # if dist.get_rank() == 0:
                #     breakpoint()

                if not samples_rewards.get(num_steps, 0):
                    samples_rewards[num_steps] = [{"rewards": rewards}]
                else:
                    samples_rewards[num_steps].append({"rewards": rewards})
                
                var_inf_images[num_steps] = images
            
            # sample
            with torch.no_grad(),autocast():
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
            denoised_latents = torch.stack(denoised_latents, dim=1) # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)

            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {   
                    "prompt": prompts,
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "denoised_latents": denoised_latents,
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
            # if config.use_regularization:
            #     sample["rewards_regularized"] = sample["rewards"] - config.penalty_constant / 2.0 * sample["regularization"]

        image_sample_rewards = {}
        for num_steps in var_inf_steps:
            counter = 0
            for sample in tqdm(
                samples_rewards[num_steps],
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                # if config.reward_fn == 'imagereward':
                #     rewards = sample["rewards"].result()
                # else:    
                rewards, reward_metadata = sample["rewards"].result()
                # accelerator.print(reward_metadata)
                sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
                
                if counter == len(samples_rewards[num_steps])-1:
                    image_sample_rewards[num_steps] = rewards
                counter += 1
        
        # Training the Value Function phase
        value_function.train()

        for inner_epoch in range(config.train.num_inner_epochs):
            total_value_loss = 0
            counts = 0
            value_optimizer.zero_grad()

            # Calculate total number of iterations for the progress bar
            total_iters = sum(len(decompose_and_batch_samples_list(samples[i:i+1], config.pretrain_VN.batch_size)) * config.v_step 
                            for i in range(len(samples)))
            
            # Create progress bar
            pbar = tqdm(total=total_iters, 
                       desc=f"Inner Epoch {inner_epoch+1}/{config.train.num_inner_epochs}: Training Value Function",
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
                            
                        value_optimizer.step()
                        value_optimizer.zero_grad()

                        current_loss = value_loss.detach().item()
                        total_value_loss += current_loss
                        counts += 1
                        
                        pbar.set_postfix({
                            'loss': f'{current_loss:.4f}', 
                            'avg_loss': f'{(total_value_loss/counts):.4f}'
                        })
                        pbar.update(1)
                        
                        del value_loss
            # value_opt_scheduler.step()

            # Logging
            avg_loss = total_value_loss / counts
            logger.info(f"Epoch={epoch}, Inner={inner_epoch}, Loss={avg_loss:.4f}")
            if accelerator.is_main_process:
                accelerator.log({"value_loss": avg_loss}, step=global_step)
            global_step += 1

        value_function.eval()

        torch.cuda.empty_cache()

        # assert config.sample.batch_size % config.sample.sub_batch_size == 0

        for sample in tqdm(
            samples,
            desc="Computing Advantage Function",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            
            # sample["VN_grad"] = inference_value_cont_batch(value_function, sample, pipeline, accelerator, config, sample_neg_prompt_embeds, config.sample.sub_batch_size)
            sample["advantages"] = inference_advantage_batch(value_function, sample, pipeline, accelerator, config)
            torch.cuda.empty_cache()
                        
        # Start training of diffusion models
        
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys() if k != "prompt"}
        
        for num_steps in var_inf_steps:
            assert len(samples_rewards[num_steps]) > 0, f"samples_rewards[{num_steps}] list is empty!"
            samples_rewards[num_steps] = {k: torch.cat([s[k] for s in samples_rewards[num_steps]]) for k in samples_rewards[num_steps][0].keys() if k != "prompt"}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )
        
        for num_steps in var_inf_steps:
            with tempfile.TemporaryDirectory() as tmpdir:

                for i, image in enumerate(var_inf_images[num_steps]):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                
                accelerator.log(
                    {
                        f"images_{num_steps}": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                            for i, (prompt, reward) in enumerate(zip(prompts, image_sample_rewards[num_steps]))  # only log rewards from process 0
                        ],
                    },
                    step=global_step,
                )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        rewards_var = {}
        for num_steps in var_inf_steps:
            rewards_var[num_steps] = accelerator.gather(samples_rewards[num_steps]["rewards"]).cpu().numpy()


        # log rewards and images
        accelerator.log(
            {"reward": rewards, "epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )

        for num_steps in var_inf_steps:
            accelerator.log(
                {
                f"reward/steps{num_steps}": rewards_var[num_steps],
                "epoch": epoch, 
                f"reward_mean/steps{num_steps}": rewards_var[num_steps].mean(), 
                f"reward_std/steps{num_steps}": rewards_var[num_steps].std(), 
                "num_steps": num_steps,
                },
                step=global_step,
             )

        samples["advantages"] = (samples["advantages"] - samples["advantages"].mean(dim=0, keepdim=True)) / (samples["advantages"].std(dim=0, keepdim=True) + 1e-8)

        del samples["rewards"]
        del samples["denoised_latents"]
        # del samples["rewards_regularized"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            # Shuffle only tensor-based entries
            samples = {k: v[perm] for k, v in samples.items() if k != "prompt"}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs", "advantages"]:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}
                
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
              
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):  
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
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
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )
                        
                        # ppo logic
                        advantages = torch.clamp(
                             sample["advantages"][:,j], -config.train.adv_clip_max, config.train.adv_clip_max)
                        
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        unet_optimizer.step()
                        unet_optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
