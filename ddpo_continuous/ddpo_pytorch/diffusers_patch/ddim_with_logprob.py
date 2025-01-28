# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the UNet prediction.
# - Instead of `variance_noise`, it takes `prev_sample` as an optional argument. If `prev_sample` is provided,
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.

from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from torchvision import transforms



def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

    """
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob

def ddim_step_with_predicted_sample_and_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )

    beta_prod_t = 1 - alpha_prod_t

    with torch.no_grad():
        # Compute predicted original sample and epsilon
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample

        # Clip or threshold predicted x_0
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # Compute variance
        variance = _get_variance(self, timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

        if use_clipped_model_output:
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # Compute direction and previous sample mean
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        prev_sample_mean = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # Compute log probability with gradients enabled
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), pred_original_sample.type(sample.dtype), log_prob


def ddim_step_with_value_grad(
    self: DDIMScheduler,
    timestep: int,
    sample: torch.FloatTensor,
    pipeline,
    embeds,
    value_function: torch.nn.Module,
    prompts,
    config,
    eta: float = 1.0,
    use_clipped_model_output: bool = False,
) -> torch.FloatTensor:
    """
    Compute value function gradients with respect to latents, incorporating noise prediction
    and handling both CFG and non-CFG cases.
    
    Args:
        self: DDIMScheduler instance
        timestep: current discrete timestep in the diffusion chain
        sample: current instance of sample being created by diffusion process
        pipeline: diffusion pipeline containing UNet and VAE
        embeds: text embeddings for conditioning
        value_function: the value function model
        prompts: text prompts or embeddings for value function
        config: configuration object containing training and sampling settings
        eta: weight of noise for added noise in diffusion step
        use_clipped_model_output: whether to use clipped model output
        
    Returns:
        torch.FloatTensor: Gradients of value function with respect to input latents
    """
    sample.requires_grad_(True)
    
    # Handle conditional guidance scale (CFG)
    if config.train.cfg:
        # Duplicate latents
        sub_latents = torch.cat([sample] * 2)
        sub_timesteps = torch.cat([timestep] * 2)
        
        # Handle embeddings properly based on type
        if isinstance(embeds, torch.Tensor):
            # For tensor embeddings, duplicate each embedding
            batch_size = sample.shape[0]
            sub_embeds = torch.cat([embeds] * 2)
        else:
            # For list/tuple embeddings (like ["", prompt]), duplicate for each sample
            batch_size = sample.shape[0]
            text_embeds = embeds[1]  # Assuming embeds[1] is the text embedding
            null_embeds = embeds[0]  # Assuming embeds[0] is the null/unconditional embedding
            
            if isinstance(text_embeds, torch.Tensor):
                sub_embeds = torch.cat([null_embeds.expand(batch_size, -1, -1), 
                                      text_embeds.expand(batch_size, -1, -1)])
            else:
                # If embeddings are not tensors, expand lists
                sub_embeds = [null_embeds] * batch_size + [text_embeds] * batch_size
        
        # Get noise prediction
        noise_pred = pipeline.unet(
            sub_latents,
            sub_timesteps,
            sub_embeds,
        ).sample
        
        # Split predictions and apply guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        model_output = noise_pred_uncond + config.sample.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        # Direct noise prediction without CFG
        model_output = pipeline.unet(
            sample,
            timestep,
            embeds if isinstance(embeds, torch.Tensor) else embeds,
        ).sample

    # Get alphas and betas for current timestep
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # Compute predicted original sample (denoised)
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output

    # Clip if needed
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
    
    # Prepare image transform
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])

    # Process current and denoised images
    current_images = pipeline.vae.decode(sample / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    current_images = (current_images + 1) / 2
    current_images = current_images.clamp(0, 1)
    current_images = resize_transform(current_images)

    denoised_images = pipeline.vae.decode(pred_original_sample / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    denoised_images = (denoised_images + 1) / 2
    denoised_images = denoised_images.clamp(0, 1)
    denoised_images = resize_transform(denoised_images)

    torch.cuda.empty_cache()

    # Compute value function
    values = value_function(
        current_images,
        denoised_images,
        prompts,
        timestep.expand(sample.shape[0])
    )

    # Compute gradients and explicitly prevent future gradient computation
    value_grad = torch.autograd.grad(
        values.sum(),
        sample,
        create_graph=False,
        retain_graph=False
    )[0]
    
    sample.requires_grad_(False)
    
    return value_grad