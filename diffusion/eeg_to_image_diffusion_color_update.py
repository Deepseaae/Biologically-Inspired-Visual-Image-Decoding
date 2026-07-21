# eeg_to_image_diffusion_local_without_mlp.py
"""
EEG -> Image Diffusion local script (no MLP projector version) - Multi-color space fusion version

Uses multiple IP-Adapters + dynamic scales to achieve color fusion depending on timestep.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import torch.nn.functional as F
import math

sys.path.append(str(Path(__file__).parent))

from custompipe_color_space import Generator4Embeds, LocalModelManager
from diffusion_prior import DiffusionPriorUNet, Pipe
from torch.utils.data import Dataset, DataLoader
from config_manager import MODEL_PATHS


class MultiColorEEGDataset(Dataset):
    """Dataset loading EEG features from four color spaces"""
    def __init__(self, eeg_rgb_path, eeg_lms_path, eeg_dkl_path, eeg_cieluv_path,
                 image_features_path=None, transform=None, mode='train'):
        # Load RGB features
        self.eeg_rgb = torch.load(eeg_rgb_path)
        if isinstance(self.eeg_rgb, dict):
            self.eeg_rgb = self.eeg_rgb.get('eeg_embeddings', self.eeg_rgb)
        if isinstance(self.eeg_rgb, np.ndarray):
            self.eeg_rgb = torch.from_numpy(self.eeg_rgb).float()
        
        # Load LMS features
        self.eeg_lms = torch.load(eeg_lms_path)
        if isinstance(self.eeg_lms, dict):
            self.eeg_lms = self.eeg_lms.get('eeg_embeddings', self.eeg_lms)
        if isinstance(self.eeg_lms, np.ndarray):
            self.eeg_lms = torch.from_numpy(self.eeg_lms).float()
        
        # Load DKL features
        self.eeg_dkl = torch.load(eeg_dkl_path)
        if isinstance(self.eeg_dkl, dict):
            self.eeg_dkl = self.eeg_dkl.get('eeg_embeddings', self.eeg_dkl)
        if isinstance(self.eeg_dkl, np.ndarray):
            self.eeg_dkl = torch.from_numpy(self.eeg_dkl).float()
        
        # Load CIELUV features
        self.eeg_cieluv = torch.load(eeg_cieluv_path)
        if isinstance(self.eeg_cieluv, dict):
            self.eeg_cieluv = self.eeg_cieluv.get('eeg_embeddings', self.eeg_cieluv)
        if isinstance(self.eeg_cieluv, np.ndarray):
            self.eeg_cieluv = torch.from_numpy(self.eeg_cieluv).float()
        
        # Ensure the number of samples is consistent
        n_samples = len(self.eeg_rgb)
        assert len(self.eeg_lms) == n_samples and len(self.eeg_dkl) == n_samples and len(self.eeg_cieluv) == n_samples, \
            "All EEG feature sample counts must be equal"
        
        # Image features
        if image_features_path:
            self.img_features = torch.load(image_features_path)
            if isinstance(self.img_features, dict):
                self.img_features = self.img_features.get('image_features', self.img_features)
            if isinstance(self.img_features, np.ndarray):
                self.img_features = torch.from_numpy(self.img_features).float()
            assert len(self.img_features) == n_samples, "Image features must have same number of samples as EEG"
        else:
            self.img_features = None
        
        print(f"[{mode.upper()}] Loaded {n_samples} samples, EEG feature dimension: {self.eeg_rgb.shape[1]}")
    
    def __len__(self):
        return len(self.eeg_rgb)
    
    def __getitem__(self, idx):
        sample = {
            'eeg_rgb': self.eeg_rgb[idx],
            'eeg_lms': self.eeg_lms[idx],
            'eeg_dkl': self.eeg_dkl[idx],
            'eeg_cieluv': self.eeg_cieluv[idx],
        }
        if self.img_features is not None:
            sample['image_feature'] = self.img_features[idx]
        return sample


class EEGToImageDiffusionLocal:
    """EEG feature to image diffusion model using local models (multi-color space fusion version)"""
    
    def __init__(self, device='cuda', latent_dim=1024,
                 prior_checkpoint=None, generator_checkpoint=None,
                 sdxl_local_path=None, ip_adapter_local_path=None,
                 modality='image'):
        
        self.device = device
        self.latent_dim = latent_dim
        self.modality = modality

        # Check local models
        print("Checking local models...")
        model_manager = LocalModelManager()
        model_manager.check_models()

        # Initialize diffusion prior model (still conditioned on RGB-EEG)
        print("Initializing diffusion prior model...")
        self.diffusion_prior = DiffusionPriorUNet(
            embed_dim=latent_dim,
            cond_dim=latent_dim,
            hidden_dim=[1024, 512, 256, 128, 64]
        ).to(device)

        # Load pretrained weights
        if prior_checkpoint and os.path.exists(prior_checkpoint):
            print(f"Loading diffusion prior weights: {prior_checkpoint}")
            self.diffusion_prior.load_state_dict(torch.load(prior_checkpoint, map_location=device))

        # Initialize diffusion pipeline
        self.pipe = Pipe(self.diffusion_prior, device=device, modality=modality)

        # Initialize image generator (multiple IP-Adapters)
        print("Initializing local image generator (multi IP-Adapter)...")
        self.image_generator = Generator4Embeds(
            # Adjusted parameters suitable for multi-color space input
            num_inference_steps=4,
            device=device,
            sdxl_local_path=sdxl_local_path,
            ip_adapter_local_path=ip_adapter_local_path,
            num_adapters=4  # RGB, LMS, DKL, CIELUV
        )
        
        # Parameters aligned with original code
        self.feature_scale = 10.0
        self.condition_dropout_prob = 0.1

        # Add optional projection layers for LMS, DKL, CIELUV (mapping EEG features to image feature space)
        # Initialized as identity mapping (trainable)
        self.proj_lms = nn.Linear(latent_dim, latent_dim, bias=False)
        self.proj_dkl = nn.Linear(latent_dim, latent_dim, bias=False)
        self.proj_cieluv = nn.Linear(latent_dim, latent_dim, bias=False)
        # Initialize with identity matrix
        nn.init.eye_(self.proj_lms.weight)
        nn.init.eye_(self.proj_dkl.weight)
        nn.init.eye_(self.proj_cieluv.weight)
        self.proj_lms.to(device)
        self.proj_dkl.to(device)
        self.proj_cieluv.to(device)

    def validate_aligned_multicolor(self, val_loader):
        """Validation (multi-color space version)"""
        was_training = self.diffusion_prior.training
        
        self.diffusion_prior.eval()
        self.proj_lms.eval()
        self.proj_dkl.eval()
        self.proj_cieluv.eval()
        
        val_losses = []
        val_sims = []
        val_proj_losses = []

        with torch.no_grad():
            for batch in val_loader:
                # Use correct key names
                eeg_rgb = batch['eeg_rgb'].to(self.device)
                lms = batch['eeg_lms'].to(self.device)
                dkl = batch['eeg_dkl'].to(self.device)
                cieluv = batch['eeg_cieluv'].to(self.device)
                img_features = batch['image_feature'].to(self.device)

                # Generate image features (conditioned on RGB)
                generated_features = self.generate_from_eeg_aligned(
                    eeg_rgb, num_inference_steps=50, guidance_scale=5.0)

                # Scale ground truth image features by 10 to maintain consistent scale
                img_features_scaled = img_features * self.feature_scale
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    generated_features,
                    img_features_scaled,
                    dim=1
                ).mean().item()
                val_sims.append(cos_sim)

                # Compute MSE loss
                mse_loss = torch.nn.functional.mse_loss(
                    generated_features,
                    img_features_scaled
                ).item()
                val_losses.append(mse_loss)
                
                # Validate projection layers
                lms_proj = self.proj_lms(lms)
                dkl_proj = self.proj_dkl(dkl)
                cieluv_proj = self.proj_cieluv(cieluv)
                
                proj_loss = (
                    F.mse_loss(lms_proj, img_features_scaled).item() +
                    F.mse_loss(dkl_proj, img_features_scaled).item() +
                    F.mse_loss(cieluv_proj, img_features_scaled).item()
                ) / 3.0
                val_proj_losses.append(proj_loss)

        # Restore mode
        if was_training:
            self.diffusion_prior.train()
            self.proj_lms.train()
            self.proj_dkl.train()
            self.proj_cieluv.train()
        
        return np.mean(val_losses), np.mean(val_sims), np.mean(val_proj_losses)

    def _train_diffusion_step_aligned(self, c_embeds, h_embeds, optimizer, lr_scheduler):
            optimizer.zero_grad()
            batch_size = h_embeds.shape[0]
            
            # Ensure dimension matching
            if c_embeds is not None and c_embeds.dim() > 2:
                c_embeds = c_embeds.view(c_embeds.shape[0], -1)
            if h_embeds.dim() > 2:
                h_embeds = h_embeds.view(h_embeds.shape[0], -1)
            
            # 1. Scale target features (maintain numerical stability)
            h_embeds_scaled = h_embeds * self.feature_scale
            
            # 2. Sample random timesteps
            timesteps = torch.randint(
                0, self.pipe.scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()
            
            # 3. Add noise (standard diffusion process)
            noise = torch.randn_like(h_embeds_scaled)
            noisy_h_embeds = self.pipe.scheduler.add_noise(h_embeds_scaled, noise, timesteps)
            
            # 4. Randomly drop condition (for classifier-free guidance)
            if torch.rand(1) < self.condition_dropout_prob:
                c_embeds = None
            
            # 5. Predict noise (core task of diffusion model)
            noise_pred = self.diffusion_prior(noisy_h_embeds, timesteps, c_embeds)
            
            # 6. Compute loss (MSE between predicted noise and true noise)
            loss = F.mse_loss(noise_pred, noise)
            
            # 7. Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            return loss.item()

    def _compute_diffusion_loss(self, c_embeds, h_embeds):
        """Compute diffusion loss (does not perform optimization step, only returns loss value)"""
        batch_size = h_embeds.shape[0]
        
        # Ensure dimension matching
        if c_embeds is not None and c_embeds.dim() > 2:
            c_embeds = c_embeds.view(c_embeds.shape[0], -1)
        if h_embeds.dim() > 2:
            h_embeds = h_embeds.view(h_embeds.shape[0], -1)
        
        # Scale target features
        h_embeds_scaled = h_embeds * self.feature_scale
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.pipe.scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise
        noise = torch.randn_like(h_embeds_scaled)
        noisy_h_embeds = self.pipe.scheduler.add_noise(h_embeds_scaled, noise, timesteps)
        
        # Randomly drop condition (CFG training)
        if torch.rand(1) < self.condition_dropout_prob:
            c_embeds = None
        
        # Predict noise
        noise_pred = self.diffusion_prior(noisy_h_embeds, timesteps, c_embeds)
        
        # Return loss
        return F.mse_loss(noise_pred, noise)

    def generate_from_eeg_aligned(self, eeg_embeddings, num_inference_steps=50, guidance_scale=5.0):
        """Generate image features from EEG embeddings (conditioned on RGB)"""
        self.diffusion_prior.eval()
        with torch.no_grad():
            if isinstance(eeg_embeddings, np.ndarray):
                eeg_embeddings = torch.from_numpy(eeg_embeddings).float()
            eeg_embeddings = eeg_embeddings.to(self.device)

            generated_features = self.pipe.generate(
                c_embeds=eeg_embeddings,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        return generated_features

    def generate_image_from_eeg(self, eeg_rgb, eeg_lms, eeg_dkl, eeg_cieluv,
                               text_prompt="", num_images=1, save_dir="./generated_images",
                               show_progress=True, prefix="generated"):
        """Generate images directly from EEG embeddings of four color spaces (dynamic fusion)"""
        os.makedirs(save_dir, exist_ok=True)

        if show_progress:
            print(f"Generating image features for {len(eeg_rgb)} samples...")

        # Ensure inputs are tensors and on the correct device
        if isinstance(eeg_rgb, np.ndarray):
            eeg_rgb = torch.from_numpy(eeg_rgb).float()
        if isinstance(eeg_lms, np.ndarray):
            eeg_lms = torch.from_numpy(eeg_lms).float()
        if isinstance(eeg_dkl, np.ndarray):
            eeg_dkl = torch.from_numpy(eeg_dkl).float()
        if isinstance(eeg_cieluv, np.ndarray):
            eeg_cieluv = torch.from_numpy(eeg_cieluv).float()
            
        eeg_rgb = eeg_rgb.to(self.device)
        eeg_lms = eeg_lms.to(self.device)
        eeg_dkl = eeg_dkl.to(self.device)
        eeg_cieluv = eeg_cieluv.to(self.device)

        # 1. Generate base image features from RGB-EEG using diffusion prior (already multiplied by 10)
        img_feat_rgb_scaled = self.generate_from_eeg_aligned(eeg_rgb)
        # Divide by 10 to restore original scale
        img_feat_rgb = img_feat_rgb_scaled / self.feature_scale

        # 2. Project the other three EEG features (if needed)
        eeg_lms_proj = self.proj_lms(eeg_lms)
        eeg_dkl_proj = self.proj_dkl(eeg_dkl)
        eeg_cieluv_proj = self.proj_cieluv(eeg_cieluv)

        # 3. Build a list of four embeddings for each sample
        # Note: each embedding should be shaped as (1, D)
        batch_size = img_feat_rgb.shape[0]
        embeds_batch_list = []
        for i in range(min(batch_size, num_images)):
            embeds_list = [
                img_feat_rgb[i:i+1],          # RGB feature (generated by diffusion prior)
                eeg_lms_proj[i:i+1],           # LMS projected feature
                eeg_dkl_proj[i:i+1],           # DKL projected feature
                eeg_cieluv_proj[i:i+1]         # CIELUV projected feature
            ]
            embeds_batch_list.append(embeds_list)

        def scale_schedule(t, T=1000):
            """
            x = t/T, 0=early, 1=late
            Weight order: [RGB, LMS, DKL, CIELUV]
            """
            x = 1.0 - (t / T)

            # RGB: exponential decay from 1.0 to ≈0.5
            w_rgb = 0.5 + 0.5 * np.exp(-5 * x)

            # The other three use Gaussian peaks with the same shape, shifted right by 0.1
            d = 0.03
            w_lms    = np.exp(-((x - 0.3) ** 2) / d)
            w_dkl    = np.exp(-((x - 0.5) ** 2) / d)
            w_cieluv = np.exp(-((x - 0.7) ** 2) / d)
            
            return [w_rgb, w_lms, w_dkl, w_cieluv]

        # 5. Batch generate images
        generated_images = self.image_generator.batch_generate(
            embeds_batch_list,
            text_prompt=text_prompt,
            scale_schedule_fn=scale_schedule
        )

        # 6. Save images
        saved_images = []
        for i, img in enumerate(generated_images):
            if img is None:
                continue
            img_path = os.path.join(save_dir, f"{prefix}_{i}.png")
            img.save(img_path)
            saved_images.append(img)
            if show_progress:
                print(f"  Image saved to: {img_path}")

        return saved_images


def main():
    parser = argparse.ArgumentParser(description="EEG to Image Diffusion Model (Multi-Color Space Fusion Version)")
    # Data arguments (must specify four EEG feature files)
    parser.add_argument("--train_eeg_rgb", type=str, default=None, help="Training RGB-EEG feature file")
    parser.add_argument("--train_eeg_lms", type=str, default=None, help="Training LMS-EEG feature file")
    parser.add_argument("--train_eeg_dkl", type=str, default=None, help="Training DKL-EEG feature file")
    parser.add_argument("--train_eeg_cieluv", type=str, default=None, help="Training CIELUV-EEG feature file")
    parser.add_argument("--train_image_features", type=str, default=None, help="Training image feature file")
    
    parser.add_argument("--test_eeg_rgb", type=str, default=None, help="Test RGB-EEG feature file")
    parser.add_argument("--test_eeg_lms", type=str, default=None, help="Test LMS-EEG feature file")
    parser.add_argument("--test_eeg_dkl", type=str, default=None, help="Test DKL-EEG feature file")
    parser.add_argument("--test_eeg_cieluv", type=str, default=None, help="Test CIELUV-EEG feature file")
    parser.add_argument("--test_image_features", type=str, default=None, help="Test image feature file")
    
    # Other arguments unchanged
    parser.add_argument("--prior_checkpoint", type=str, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--text_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu, cuda, cuda:0, cuda:1, etc.")
    parser.add_argument("--sdxl_path", type=str, default=None)
    parser.add_argument("--ip_adapter_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Parameter checking omitted; ensure all four features exist.

    # Initialize model
    diffusion_model = EEGToImageDiffusionLocal(
        device=args.device,
        latent_dim=1024,  # Can be inferred from feature files
        prior_checkpoint=args.prior_checkpoint,
        sdxl_local_path=args.sdxl_path,
        ip_adapter_local_path=args.ip_adapter_path
    )

    # Training mode (joint training of diffusion prior + projection layers)
    if args.train:
        # Create dataset
        train_dataset = MultiColorEEGDataset(
            args.train_eeg_rgb, args.train_eeg_lms, args.train_eeg_dkl, args.train_eeg_cieluv,
            args.train_image_features, mode='train'
        )
        total = len(train_dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Optimizer (includes diffusion prior and projection layers)
        optimizer = torch.optim.Adam(
            list(diffusion_model.diffusion_prior.parameters()) +
            list(diffusion_model.proj_lms.parameters()) +
            list(diffusion_model.proj_dkl.parameters()) +
            list(diffusion_model.proj_cieluv.parameters()),
            lr=1e-4
        )
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=500,
            num_training_steps=len(train_loader) * args.epochs
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # ========== Training phase ==========
            diffusion_model.diffusion_prior.train()
            diffusion_model.proj_lms.train()
            diffusion_model.proj_dkl.train()
            diffusion_model.proj_cieluv.train()
            
            train_losses = []
            proj_losses = []  # Record projection losses
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
                rgb = batch['eeg_rgb'].to(args.device)
                lms = batch['eeg_lms'].to(args.device)
                dkl = batch['eeg_dkl'].to(args.device)
                cieluv = batch['eeg_cieluv'].to(args.device)
                img_feat = batch['image_feature'].to(args.device)
                
                # ---- Diffusion prior training (conditioned on RGB) ----
                optimizer.zero_grad()
                
                # 1. Diffusion loss (RGB -> Image)
                loss_diffusion = diffusion_model._compute_diffusion_loss(rgb, img_feat)
                
                # 2. Projection loss (LMS/DKL/CIELUV -> Image)
                img_feat_scaled = img_feat * diffusion_model.feature_scale
                
                lms_proj = diffusion_model.proj_lms(lms)
                dkl_proj = diffusion_model.proj_dkl(dkl)
                cieluv_proj = diffusion_model.proj_cieluv(cieluv)
                
                # Projection loss: let projected features directly reconstruct image features (supervision signal)
                loss_proj_lms = F.mse_loss(lms_proj, img_feat_scaled)
                loss_proj_dkl = F.mse_loss(dkl_proj, img_feat_scaled)
                loss_proj_cieluv = F.mse_loss(cieluv_proj, img_feat_scaled)
                
                # Projection consistency loss: projected features should be consistent with each other (enhance stability)
                loss_proj_consistency = (
                    F.mse_loss(lms_proj, dkl_proj) + 
                    F.mse_loss(dkl_proj, cieluv_proj) +
                    F.mse_loss(cieluv_proj, lms_proj)
                ) / 3.0
                
                loss_proj = (loss_proj_lms + loss_proj_dkl + loss_proj_cieluv) / 3.0 + \
                        0.1 * loss_proj_consistency  # Consistency weight is small
                
                # 3. Total loss for joint optimization
                total_loss = loss_diffusion + 0.5 * loss_proj  # Projection loss weight can be adjusted
                
                total_loss.backward()

                # Clip gradients for all parameters
                torch.nn.utils.clip_grad_norm_(
                    list(diffusion_model.diffusion_prior.parameters()) +
                    list(diffusion_model.proj_lms.parameters()) +
                    list(diffusion_model.proj_dkl.parameters()) +
                    list(diffusion_model.proj_cieluv.parameters()),
                    max_norm=1.0
                )
                
                optimizer.step()
                lr_scheduler.step()
                
                train_losses.append(loss_diffusion.item())
                proj_losses.append(loss_proj.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_proj_loss = np.mean(proj_losses)
            
            # ========== Validation phase (using the new multi-color validation method) ==========
            avg_val_loss, avg_val_sim, avg_val_proj_loss = diffusion_model.validate_aligned_multicolor(val_loader)
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - Diffusion: {avg_train_loss:.6f}, Proj: {avg_proj_loss:.6f}")
            print(f"  Val   - Diffusion: {avg_val_loss:.6f}, Sim: {avg_val_sim:.4f}, Proj: {avg_val_proj_loss:.6f}")
            
            # ========== Save best model ==========
            combined_metric = avg_val_loss + 0.1 * avg_val_proj_loss
            
            if combined_metric < best_val_loss:
                best_val_loss = combined_metric
                
                # Define unified checkpoint directory path
                ckpt_dir = os.path.join(args.output_dir, "checkpoints_O")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                torch.save(diffusion_model.diffusion_prior.state_dict(),
                        os.path.join(ckpt_dir, "diffusion_prior_best.pth"))
                torch.save(diffusion_model.proj_lms.state_dict(),
                        os.path.join(ckpt_dir, "proj_lms_best.pth"))
                torch.save(diffusion_model.proj_dkl.state_dict(),
                        os.path.join(ckpt_dir, "proj_dkl_best.pth"))
                torch.save(diffusion_model.proj_cieluv.state_dict(),
                        os.path.join(ckpt_dir, "proj_cieluv_best.pth"))
                
                print(f"  -> Best model saved (combined metric={combined_metric:.6f})")
            

    # Generation mode
    if args.generate:
        # Automatically find the best checkpoint path (assuming saved under output_dir/checkpoints_O/)
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints_O")
        
        # Determine diffusion prior weight path
        if args.prior_checkpoint is None:
            prior_ckpt = os.path.join(checkpoint_dir, "diffusion_prior_best.pth")
            if os.path.exists(prior_ckpt):
                print(f"Auto-loading diffusion prior weights: {prior_ckpt}")
                diffusion_model.diffusion_prior.load_state_dict(torch.load(prior_ckpt, map_location=args.device))
            else:
                print("Warning: best diffusion prior weights not found, using randomly initialized model.")
        else:
            diffusion_model.diffusion_prior.load_state_dict(torch.load(args.prior_checkpoint, map_location=args.device))
        
        # Load projection layer weights (mandatory)
        proj_files = {
            'proj_lms': 'proj_lms_best.pth',
            'proj_dkl': 'proj_dkl_best.pth',
            'proj_cieluv': 'proj_cieluv_best.pth'
        }
        for proj_name, filename in proj_files.items():
            proj_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(proj_path):
                getattr(diffusion_model, proj_name).load_state_dict(torch.load(proj_path, map_location=args.device))
                print(f"Loaded {proj_name} weights: {proj_path}")
            else:
                print(f"Warning: {proj_name} weight file {proj_path} does not exist, using initial values.")

        # Load test data and process
        test_rgb = torch.load(args.test_eeg_rgb)
        test_lms = torch.load(args.test_eeg_lms)
        test_dkl = torch.load(args.test_eeg_dkl)
        test_cieluv = torch.load(args.test_eeg_cieluv)
        
        # Handle possible dict format
        if isinstance(test_rgb, dict):
            test_rgb = test_rgb.get('eeg_embeddings', test_rgb)
        if isinstance(test_lms, dict):
            test_lms = test_lms.get('eeg_embeddings', test_lms)
        if isinstance(test_dkl, dict):
            test_dkl = test_dkl.get('eeg_embeddings', test_dkl)
        if isinstance(test_cieluv, dict):
            test_cieluv = test_cieluv.get('eeg_embeddings', test_cieluv)
            
        # Convert to tensor
        if isinstance(test_rgb, np.ndarray):
            test_rgb = torch.from_numpy(test_rgb).float()
        if isinstance(test_lms, np.ndarray):
            test_lms = torch.from_numpy(test_lms).float()
        if isinstance(test_dkl, np.ndarray):
            test_dkl = torch.from_numpy(test_dkl).float()
        if isinstance(test_cieluv, np.ndarray):
            test_cieluv = torch.from_numpy(test_cieluv).float()
        
        num_samples = min(args.num_samples, len(test_rgb))
        selected_rgb = test_rgb[:num_samples]
        selected_lms = test_lms[:num_samples]
        selected_dkl = test_dkl[:num_samples]
        selected_cieluv = test_cieluv[:num_samples]
        
        diffusion_model.generate_image_from_eeg(
            selected_rgb, selected_lms, selected_dkl, selected_cieluv,
            text_prompt=args.text_prompt,
            num_images=num_samples,
            save_dir=os.path.join(args.output_dir, "generated"),
            prefix="multi"
        )


if __name__ == "__main__":
    main()