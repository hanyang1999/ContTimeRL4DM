import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from scripts.src.models.BLIP.blip_pretrain import BLIP_Pretrain

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)

class ImageRewardValue(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        
        # Original BLIP and MLP for latents (frozen)
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.mlp = MLP(768)

        self.mlp.eval()
        
        # New BLIP and MLP for denoised latents (learnable)
        self.denoised_blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.denoised_mlp = MLP(768)
        
        # self.preprocess = _transform(224)
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072
        
        # Freeze original BLIP and MLP
        # self.freeze_original_models()
    
    def train(self, mode=True):
        """Override train method to keep original MLP in eval mode"""
        super().train(mode)
        # Force original MLP to stay in eval mode regardless of model's training mode
        self.mlp.eval()
        return self

    def eval(self):
        """Override eval method"""
        super().eval()
        # Ensure original MLP stays in eval mode
        self.mlp.eval()
        return self

    def skip_func(self, timesteps):
        return torch.cos((torch.pi / 2) * (timesteps-1)/1000)
    
    def freeze_original_models(self):
        """Freeze the original BLIP and MLP models"""
        for param in self.blip.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        self.mlp.eval()
            
    def freeze_denoised_layers(self, fix_rate=0.5):
        """Freeze layers up to fix_rate in denoised BLIP."""
        if fix_rate <= 0:
            return
            
        # Get number of layers by counting blocks in visual encoder
        image_layer_num = len(self.denoised_blip.visual_encoder.blocks)
        text_fix_num = f"layer.{int(12 * fix_rate)}"
        image_fix_num = f"blocks.{int(image_layer_num * fix_rate)}"
        
        # Freeze text encoder layers
        for name, parms in self.denoised_blip.text_encoder.named_parameters():
            parms.requires_grad_(False)
            if text_fix_num in name:
                break
                
        # Freeze visual encoder layers    
        for name, parms in self.denoised_blip.visual_encoder.named_parameters():
            parms.requires_grad_(False)
            if image_fix_num in name:
                break

    # def forward(self, images, denoised_images, prompts, timesteps=None):
    #     self.mlp.eval()

    #     cur_device = next(self.blip.parameters()).device

    #     text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(cur_device)
        
    #     # Process original images with frozen BLIP and MLP
    #     with torch.no_grad():
    #         image_embeds = self.blip.visual_encoder(images).to(cur_device)
    #         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(cur_device)

    #         text_output = self.blip.text_encoder(
    #             text_input.input_ids,
    #             encoder_hidden_states=image_embeds,
    #             encoder_attention_mask=image_atts,
    #             return_dict=True,
    #         )
            
    #         combined_embedding = text_output.last_hidden_state[:, 0, :].float()
    #         value = self.mlp(combined_embedding)
    #         value = (value - self.mean) / self.std

    #     # Process denoised images with learnable BLIP and MLP
    #     denoised_image_embeds = self.denoised_blip.visual_encoder(denoised_images).to(cur_device)
    #     denoised_image_atts = torch.ones(denoised_image_embeds.size()[:-1], dtype=torch.long).to(cur_device)
        
    #     denoised_text_output = self.denoised_blip.text_encoder(
    #         text_input.input_ids,
    #         encoder_hidden_states=denoised_image_embeds,
    #         encoder_attention_mask=denoised_image_atts,
    #         return_dict=True,
    #     )
        
    #     denoised_combined_embedding = denoised_text_output.last_hidden_state[:, 0, :].float()
    #     denoised_value = self.denoised_mlp(denoised_combined_embedding)
    #     denoised_value = (denoised_value - self.mean) / self.std
        
    #     return self.skip_func(timesteps) * value.squeeze(-1) + (1-self.skip_func(timesteps)) * denoised_value.squeeze(-1)

    def forward(self, images, denoised_images, prompts, timesteps=None):
        self.mlp.eval()

        cur_device = next(self.blip.parameters()).device

        text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(cur_device)
        
        # Process original images with frozen BLIP and MLP
        image_embeds = self.blip.visual_encoder(denoised_images).to(cur_device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(cur_device)

        text_output = self.blip.text_encoder(
            text_input.input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        combined_embedding = text_output.last_hidden_state[:, 0, :].float()
        value = self.mlp(combined_embedding)
        value = (value - self.mean) / self.std

        # Process denoised images with learnable BLIP and MLP
        denoised_image_embeds = self.denoised_blip.visual_encoder(images).to(cur_device)
        denoised_image_atts = torch.ones(denoised_image_embeds.size()[:-1], dtype=torch.long).to(cur_device)
        
        denoised_text_output = self.denoised_blip.text_encoder(
            text_input.input_ids,
            encoder_hidden_states=denoised_image_embeds,
            encoder_attention_mask=denoised_image_atts,
            return_dict=True,
        )
        
        denoised_combined_embedding = denoised_text_output.last_hidden_state[:, 0, :].float()
        denoised_value = self.denoised_mlp(denoised_combined_embedding)
        denoised_value = (denoised_value - self.mean) / self.std
        
        return self.skip_func(timesteps) * value.squeeze(-1) + (1-self.skip_func(timesteps)) * denoised_value.squeeze(-1)