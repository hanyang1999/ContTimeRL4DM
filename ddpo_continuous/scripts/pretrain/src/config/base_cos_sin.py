from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    config.seed = 42

    config.design = {"first":"denoised","second":"original","skip_func_in":"cos","skip_func_out":"sin"}

    # Basic training settings
    config.num_epochs = 100
    config.save_freq = 20
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.logdir = "pretrain_ckpts"
    # config.run_name = "image_reward_training_fixrate=0.7_network=flipped_warmup=0.05"
    config.run_name = "image_reward_training_fixrate=0.7_cos-sin_armup=0.05"
    config.use_lora = False
    
    # Model settings
    config.pretrained = config_dict.ConfigDict()
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = None
    
    # Value network settings
    config.value_network = config_dict.ConfigDict()
    config.value_network.med_config = "config/med_config.json"
    
    # Training parameters
    config.pretrain_VN = config_dict.ConfigDict()
    config.pretrain_VN.batch_size = 100
    config.pretrain_VN.learning_rate = 1e-7
    config.pretrain_VN.adam_beta1 = 0.9
    config.pretrain_VN.adam_beta2 = 0.999
    config.pretrain_VN.adam_weight_decay = 0.01
    config.pretrain_VN.adam_epsilon = 1e-8
    config.pretrain_VN.gradient_accumulation_steps = 32
    config.pretrain_VN.num_inner_epochs = 1
    config.pretrain_VN.max_grad_norm = 1.0
    config.pretrain_VN.warmup_ratio = 0.05
    
    # Sampling parameters
    config.sample = config_dict.ConfigDict()
    config.sample.batch_size = 64
    config.sample.num_batches_per_epoch = 1
    config.sample.num_steps = 50
    config.sample.guidance_scale = 7.5
    config.sample.eta = 1.0
    
    # Training steps
    config.v_step = 3
    
    
    # Prompt and reward settings
    config.prompt_fn = "single_prompt"
    config.prompt_fn_kwargs = {}
    # config.prompt_fn_kwargs = {"prompts": ["a photo of a cat", "a photo of a dog"]}
    config.reward_fn = "imagereward"
    
    return config
