import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.nn import functional as F
import math 
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EEG_Encoder(nn.Module):
    def __init__(self, z_dim=1024, c_num=17, timesteps=(0, 250), num_subjects=10, drop_proj=0.3,
                 use_attention=True, use_subject_linear=True, use_cogcap=True):
        super(EEG_Encoder, self).__init__()
        self.z_dim = z_dim  # output dimension
        self.c_num = c_num  # number of channels
        self.timesteps = timesteps  # time range
        
        # Calculate the actual number of time steps used
        self.time_len = timesteps[1] - timesteps[0]
        
        if use_cogcap:
            # Use Cogcap-style encoder
            self.eeg_encoder = SimpleCogcap(
                num_channels=c_num,
                sequence_length=self.time_len,
                num_subjects=num_subjects,
                num_latents=z_dim
            )
        else:
            # Multi-layer perceptron
            self.input_dim = c_num * self.time_len
            self.eeg_encoder = nn.Sequential(
                nn.Linear(self.input_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(drop_proj),
                nn.Linear(2048, z_dim),
                nn.LayerNorm(z_dim)
            )
        
        # Used for dimension adjustment or further processing
        if z_dim != 1024 and use_cogcap:
            self.final_proj = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.GELU(),
                nn.Dropout(drop_proj),
                nn.LayerNorm(z_dim)
            )
        else:
            self.final_proj = nn.Identity()
        
        # Contrastive learning parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: EEG input, shape [batch_size, channels, timesteps]
        Returns:
            Projected features, shape [batch_size, z_dim]
        """
        batch_size = x.shape[0]
        
        # 1. Time cropping
        start_idx, end_idx = self.timesteps
        if start_idx != 0 or end_idx != x.shape[2]:
            x = x[:, :, start_idx:end_idx]
        
        # 2. EEG encoding
        if isinstance(self.eeg_encoder, SimpleCogcap):
            encoded = self.eeg_encoder(x)  # Use Cogcap encoder
        else:
            # Flatten input
            x_flat = x.reshape(batch_size, -1)
            encoded = self.eeg_encoder(x_flat)  # Use simple MLP
        
        # 3. Final projection
        output = self.final_proj(encoded)
        
        return output
    
    def get_logit_scale(self):
        """Get logit scale for contrastive learning"""
        return self.logit_scale.exp()
    
    def compute_contrastive_loss(self, eeg_features, image_features):
        """Compute contrastive loss"""
        if hasattr(self.eeg_encoder, 'loss_func'):
            return self.eeg_encoder.loss_func(eeg_features, image_features)
        else:
            # Use contrastive loss
            eeg_features = F.normalize(eeg_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * eeg_features @ image_features.t()
            labels = torch.arange(eeg_features.shape[0], device=eeg_features.device)
            return F.cross_entropy(logits, labels)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1).to(x.device)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model
    
    def forward(self, src):
        src = src.permute(2, 0, 1)  # [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # [batch_size, channel, time_length]

class SimpleCogcap(nn.Module):
    """Cogcap encoder"""
    # Set default num_channels to 17
    def __init__(self, num_channels=17, sequence_length=250, num_subjects=10, num_latents=1024):
        super(SimpleCogcap, self).__init__()
        
        # Attention module
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        
        # Subject-specific linear layers
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length) for _ in range(min(1, num_subjects))]
        )
        
        # EEG encoding and projection
        # Pass channels parameter to Enc_eeg
        self.enc_eeg = Enc_eeg(channels=num_channels)
        self.proj_eeg = Proj_eeg(embedding_dim=1440, proj_dim=num_latents)
        # self.proj_eeg = Proj_eeg(embedding_dim=217360, proj_dim=num_latents)
        
        # Contrastive learning parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    
    def forward(self, x):
        # x: [batch, channels, timesteps]
        
        # Attention processing
        if hasattr(self, 'attention_model'):
            x = self.attention_model(x)
        
        # Subject-specific transformation (use only the first one)
        if len(self.subject_wise_linear) > 0:
            x = self.subject_wise_linear[0](x)
        
        # EEG encoding
        eeg_embedding = self.enc_eeg(x)  # [batch, 1440]
        
        # Project to target dimension
        out = self.proj_eeg(eeg_embedding)  # [batch, num_latents]
        
        return out

class PatchEmbedding(nn.Module):
    # Add channels parameter, default changed to 17
    def __init__(self, emb_size=40, channels=17):
        super().__init__()
        # PatchEmbedding
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # After changing channels, this also needs to be updated: use the passed channels variable
            nn.Conv2d(40, 40, (channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, channels, 250]
        x = self.tsconv(x)  # [batch, 40, 1, 36]
        x = x.squeeze(2)    # [batch, 40, 36]
        return x

class Enc_eeg(nn.Module):
    # Add channels parameter
    def __init__(self, emb_size=40, channels=17):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, channels=channels)
        
    def forward(self, x):
        # x: [batch, channels, 250]
        x = self.patch_embedding(x)  # [batch, 40, 36]
        # Flatten
        x = x.contiguous().view(x.size(0), -1)  # [batch, 1440]
        return x

class Proj_eeg(nn.Module):
    # def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.1):
    def __init__(self, embedding_dim=217360, proj_dim=1024, drop_proj=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    
    def forward(self, x):
        return self.model(x)

class ClipLoss(nn.Module):
    """CLIP contrastive loss"""
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Create labels
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Compute loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss


class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.5):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        tmin, tmax = timesteps[0], timesteps[1]
            
        self.input_dim = self.c_num * (tmax - tmin)


        self.encoder = nn.Sequential(
            # Use nn.LazyLinear instead of nn.Linear(self.input_dim, 1024)
            # It is PyTorch's magic that can automatically infer input dimension when first receiving data!
            nn.Linear(self.input_dim, 1024), 
            nn.GELU(),
            nn.Dropout(drop_proj),
            nn.LayerNorm(1024),
                       
            ResidualAdd(nn.Sequential(
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Dropout(drop_proj),
                nn.LayerNorm(1024),
            )),
            
            nn.Linear(1024, z_dim),
            nn.LayerNorm(z_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Use -1 to let PyTorch automatically flatten the remaining dimensions (i.e., channels * time)
        # Using reshape avoids view errors due to non-contiguous memory
        x = x.reshape(x.shape[0], -1)
        return self.encoder(x)
    
    def get_logit_scale(self):
        return self.softplus(self.logit_scale)

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        
class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        
class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )