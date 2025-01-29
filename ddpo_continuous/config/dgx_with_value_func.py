import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():

    print("Loading config from dgx_continuous.py")
    
    config = base.get_config()
    
    config.score_fixed = False
    
    config.run_name = "ddpo_compressibility"
    
    config.use_regularization = False
    
    config.penalty_constant = 0.0001

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    
    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 20
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.

    # the DGX machine I used had 7 GPUs, so this corresponds to 7 * 6 * 4 = 256 samples per epoch.
    config.sample.batch_size = 6 #8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (6 * 4) / (3 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 3
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    return config

def imagereward():
    config = compressibility()

    config.save_freq = 20
    
    #config.run_name = "ddpo_incompressibility"

    config.num_epochs = 100
    config.reward_fn = "imagereward"

    config.sample.batch_size = 64
    config.sample.num_batches_per_epoch = 2

    config.train.clip_range = 1e-5
    config.train.learning_rate = 3e-5

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 16

    # config.prompt_fn = "activities"
    # config.prompt_fn = "simple_animal"
    config.prompt_fn = "single_prompt"
    # config.per_prompt_stat_tracking = {
    #     "buffer_size": 32,
    #     "min_count": 16,
    # }

    # Model settings
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = None

    config.value_network = ml_collections.ConfigDict()
    config.value_network.med_config = "config/med_config.json"
    config.design = {"first":"denoised","second":"original","skip_func_in":"cos","skip_func_out":"sin"}
    
    # Training parameters
    config.pretrain_VN = ml_collections.ConfigDict()
    config.pretrain_VN.batch_size = 100
    config.pretrain_VN.learning_rate = 1e-7
    config.pretrain_VN.adam_beta1 = 0.9
    config.pretrain_VN.adam_beta2 = 0.999
    config.pretrain_VN.adam_weight_decay = 0.01
    config.pretrain_VN.adam_epsilon = 1e-8
    config.pretrain_VN.gradient_accumulation_steps = 32
    config.pretrain_VN.num_inner_epochs = 1
    config.pretrain_VN.max_grad_norm = 1.0
    config.pretrain_VN.warmup_ratio = 0.1
    config.pretrained_VN_path ="/root/filesystem/gpu_8/ContTimeRL4DM/ddpo_continuous/scripts/pretrain/src/pretrain_ckpts/image_reward_training_fixrate=0.7_cos-sin_warmup=0.05_2025.01.29_13.07.45/value_network_epoch_60.pt"

    config.v_step = 1
    
    config.run_name = f"ddpo_cont_no_regularization_lr={config.train.learning_rate}_clip={config.train.clip_range}_seed={config.seed}"
    return config

def get_config(name):
    return globals()[name]()