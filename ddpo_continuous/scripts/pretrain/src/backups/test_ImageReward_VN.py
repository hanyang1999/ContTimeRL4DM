import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from ImageReward_VN import ImageRewardValue
import ml_collections
from torchvision import transforms

def create_test_config():
    config = ml_collections.ConfigDict()
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = None
    
    config.value_network = ml_collections.ConfigDict()
    config.value_network.med_config = "config/med_config.json"
    
    config.mixed_precision = "no"
    config.use_lora = False
    return config

def test_pipeline():
    config = create_test_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=torch.float32
    ).to(device)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker = None
    
    value_function = ImageRewardValue(
        med_config=config.value_network.med_config,
        device=device
    ).to(device)
    value_function.freeze_layers(fix_rate=0.7)
    
    # Add image resizing transform
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    prompt = "a beautiful mountain landscape at sunset"
    
    with torch.no_grad():
        latents = pipeline(
            prompt,
            num_inference_steps=20,
            output_type="latent"
        ).images
        
        # Decode latents and normalize to [0,1]
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        # Resize and normalize for BLIP
        images = resize_transform(images)
        
        # Get value prediction
        value = value_function(images, [prompt])
        print(f"Predicted value: {value.item()}")

if __name__ == "__main__":
    test_pipeline()