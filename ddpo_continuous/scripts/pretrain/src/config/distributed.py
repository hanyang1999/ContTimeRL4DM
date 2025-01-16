from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()
    
    # Basic settings
    config.seed = 42
    config.num_epochs = 1000
    config.save_freq = 200
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.logdir = "pretrain_ckpts"
    config.run_name = "image_reward_training"
    config.use_lora = False
    
    # Pretrained model settings
    config.pretrained = config_dict.ConfigDict()
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = None
    
    # Value network settings
    config.value_network = config_dict.ConfigDict()
    config.value_network.med_config = "config/med_config.json"
    
    # Training parameters adjusted for small sample size
    config.train = config_dict.ConfigDict()
    config.train.batch_size = 4  # Set to match total number of samples
    config.train.learning_rate = 1e-5
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 0.01
    config.train.adam_epsilon = 1e-8
    config.train.gradient_accumulation_steps = 16
    config.train.num_inner_epochs = 1
    config.train.max_grad_norm = 1.0
    
    # Sampling parameters
    config.sample = config_dict.ConfigDict()
    config.sample.batch_size = 4  # Set to match total number of samples
    config.sample.num_batches_per_epoch = 16  # Increased to generate more samples
    config.sample.num_steps = 50
    config.sample.guidance_scale = 7.5
    config.sample.eta = 1.0
    
    # Additional training parameters
    config.v_step = 2  # Verified this is positive
    config.prompt_fn = "single_prompt"
    config.prompt_fn_kwargs = {}
    config.reward_fn = "imagereward"
    
    # Add validation to ensure proper distributed setup
    def validate_config(cfg):
        if cfg.train.batch_size < 4:
            raise ValueError("Training batch size must be at least 4 for proper distribution")
        if cfg.sample.batch_size < 4:
            raise ValueError("Sampling batch size must be at least 4 for proper distribution")
        if cfg.v_step <= 0:
            raise ValueError("v_step must be positive")
        return cfg
    
    return validate_config(config)