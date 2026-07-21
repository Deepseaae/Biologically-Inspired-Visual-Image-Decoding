import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2-Way Identification function
@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    """
    Compute 2-way identification accuracy.
    Args:
        all_brain_recons: reconstructed image tensor [N, C, H, W]
        all_images: original image tensor [N, C, H, W]
        model: feature extraction model
        preprocess: preprocessing function
        feature_layer: name of feature layer to extract
        return_avg: whether to return average accuracy
    Returns:
        If return_avg=True: average accuracy
        Otherwise: (success_count, total_comparisons)
    """
    # Preprocess all images
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    
    # Extract features
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    # Compute correlation matrix
    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]  # cross-correlations between originals and recons
    congruents = np.diag(r)  # diagonal corresponds to matching pairs

    # Check if each match is better than all others
    success = r < congruents.reshape(-1, 1)
    success_cnt = np.sum(success, axis=0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1

def load_all_images_from_paths(image_paths_file, original_root, generated_dir, max_images=982):
    """
    Load all images from path mapping file.
    Returns two tensors: original images and reconstructed images.
    """
    print("Loading image path mapping...")
    
    # Read image path mapping
    idx_to_path = {}
    with open(image_paths_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                idx = int(parts[0].strip())
                path = parts[1].strip()
                idx_to_path[idx] = path
    
    print(f"Loaded {len(idx_to_path)} image paths")
    
    # Collect image pairs
    all_original_tensors = []
    all_recon_tensors = []
    valid_indices = []
    
    print("\nLoading image data...")
    for idx in tqdm(range(min(max_images, 200))):  # Assume max 200 images
        if idx not in idx_to_path:
            continue
        
        # Build file paths
        rel_path = idx_to_path[idx]
        original_path = original_root / rel_path
        generated_path = generated_dir / f"generated_{idx}.png"
        
        if not original_path.exists():
            print(f"Warning: original image not found - {original_path}")
            continue
        
        if not generated_path.exists():
            print(f"Warning: generated image not found - {generated_path}")
            continue
        
        try:
            # Load original image
            orig_img = Image.open(original_path).convert('RGB')
            # Load generated image
            gen_img = Image.open(generated_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            orig_tensor = transform(orig_img).unsqueeze(0)
            gen_tensor = transform(gen_img).unsqueeze(0)
            
            all_original_tensors.append(orig_tensor)
            all_recon_tensors.append(gen_tensor)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"Failed to load image {idx}: {e}")
            continue
    
    if not all_original_tensors:
        raise ValueError("No valid image pairs found")
    
    # Concatenate all tensors
    all_images = torch.cat(all_original_tensors, dim=0).to(device)
    all_brain_recons = torch.cat(all_recon_tensors, dim=0).to(device)
    
    print(f"Successfully loaded {len(valid_indices)} image pairs")
    print(f"Image shape: {all_images.shape}")
    print(f"Reconstruction shape: {all_brain_recons.shape}")
    
    return all_images, all_brain_recons, valid_indices

def calculate_pixcorr_jupyter(all_images, all_brain_recons):
    """Compute PixCorr following Jupyter code"""
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()
    
    corrsum = 0
    for i in tqdm(range(len(all_images)), desc="Calculating PixCorr"):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    pixcorr = corrsum / len(all_images)
    
    return pixcorr

def calculate_ssim_jupyter(all_images, all_brain_recons):
    """Compute SSIM following Jupyter code"""
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim
    
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())
    
    ssim_score = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="Calculating SSIM"):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, 
                               sigma=1.5, use_sample_covariance=False, data_range=1.0))
    
    return np.mean(ssim_score)

def evaluate_with_alexnet(all_images, all_brain_recons):
    """Evaluate using AlexNet following Jupyter code"""
    from torchvision.models import alexnet, AlexNet_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    
    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(alexnet(weights=alex_weights), 
                                          return_nodes=['features.4', 'features.11']).to(device)
    alex_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    alexnet2 = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                     alex_model, preprocess, 'features.4')
    alexnet5 = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                     alex_model, preprocess, 'features.11')
    
    return np.mean(alexnet2), np.mean(alexnet5)

def evaluate_with_inception(all_images, all_brain_recons):
    """Evaluate using InceptionV3 following Jupyter code"""
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                               return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
    
    return np.mean(all_per_correct)

def evaluate_with_clip(all_images, all_brain_recons):
    """Evaluate using CLIP following Jupyter code"""
    import clip
    
    clip_model, _ = clip.load("ViT-L/14", device=device)
    
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            clip_model.encode_image, preprocess, None)
    
    return np.mean(all_per_correct)

def evaluate_with_efficientnet(all_images, all_brain_recons):
    """Evaluate using EfficientNet following Jupyter code"""
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake), -1).cpu().numpy()
    
    distances = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))])
    return distances.mean()

def evaluate_with_swav(all_images, all_brain_recons):
    """Evaluate using SwAV following Jupyter code"""
    import torch.hub
    from torchvision.models.feature_extraction import create_feature_extractor
    
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))['avgpool']
    fake = fake.reshape(len(fake), -1).cpu().numpy()
    
    distances = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))])
    return distances.mean()

def main():
    """Main evaluation function"""
    # Configure paths
    base_path = "/data2/ww/Uncertainty-aware-Blur-Prior-main"
    
    # Image path file possibilities
    possible_paths = [
        f"{base_path}/diffusion/generated_results/image_paths.txt",
        f"{base_path}/data/things-eeg/Image_set_Resize/image_paths.txt",
        "./image_paths.txt"
    ]
    
    image_paths_file = None
    for path in possible_paths:
        if Path(path).exists():
            image_paths_file = Path(path)
            break
    
    if image_paths_file is None:
        print("Error: image path file not found")
        sys.exit(1)
    
    print(f"Using image path file: {image_paths_file}")
    
    # Define directories
    generated_dir = Path(f"{base_path}/diffusion/generated_results/generated_images")
    original_root = Path(f"{base_path}/data/things-eeg/Image_set_Resize")
    
    # Load all images
    all_images, all_brain_recons, valid_indices = load_all_images_from_paths(
        image_paths_file, original_root, generated_dir
    )
    
    print("\n" + "="*80)
    print("Starting reconstruction metrics computation")
    print("="*80)
    
    # 1. PixCorr
    print("\n1. Computing PixCorr...")
    pixcorr = calculate_pixcorr_jupyter(all_images, all_brain_recons)
    print(f"   PixCorr: {pixcorr:.4f}")
    
    # 2. SSIM
    print("\n2. Computing SSIM...")
    ssim_value = calculate_ssim_jupyter(all_images, all_brain_recons)
    print(f"   SSIM: {ssim_value:.4f}")
    
    # 3. AlexNet metrics
    print("\n3. Computing AlexNet metrics...")
    alexnet2, alexnet5 = evaluate_with_alexnet(all_images, all_brain_recons)
    print(f"   AlexNet(2): {alexnet2:.4f}")
    print(f"   AlexNet(5): {alexnet5:.4f}")
    
    # 4. InceptionV3
    print("\n4. Computing InceptionV3...")
    inception = evaluate_with_inception(all_images, all_brain_recons)
    print(f"   InceptionV3: {inception:.4f}")
    
    # 5. CLIP
    print("\n5. Computing CLIP...")
    clip_value = evaluate_with_clip(all_images, all_brain_recons)
    print(f"   CLIP: {clip_value:.4f}")
    
    # 6. EfficientNet
    print("\n6. Computing EfficientNet...")
    effnet = evaluate_with_efficientnet(all_images, all_brain_recons)
    print(f"   EfficientNet Distance: {effnet:.4f}")
    
    # 7. SwAV
    print("\n7. Computing SwAV...")
    swav = evaluate_with_swav(all_images, all_brain_recons)
    print(f"   SwAV Distance: {swav:.4f}")
    
    # Summary table
    print("\n" + "="*80)
    print("Evaluation Results Summary")
    print("="*80)
    
    data = {
        "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
        "Value": [pixcorr, ssim_value, alexnet2, alexnet5, inception, clip_value, effnet, swav],
        "Description": [
            "Pixel-level correlation (higher is better)",
            "Structural similarity index (higher is better)",
            "AlexNet layer2 2-way identification rate (higher is better)",
            "AlexNet layer5 2-way identification rate (higher is better)",
            "Inception V3 2-way identification rate (higher is better)",
            "CLIP 2-way identification rate (higher is better)",
            "EfficientNet feature correlation distance (lower is better)",
            "SwAV feature correlation distance (lower is better)"
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Save results
    output_dir = Path("jupyter_style_results")
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = output_dir / f"reconstruction_metrics_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    detailed_results = {
        "timestamp": timestamp,
        "num_pairs": len(valid_indices),
        "valid_indices": valid_indices,
        "metrics": df.to_dict('records'),
        "config": {
            "device": str(device),
            "image_paths_file": str(image_paths_file),
            "original_root": str(original_root),
            "generated_dir": str(generated_dir)
        }
    }
    
    json_path = output_dir / f"detailed_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    print("\n" + "="*80)
    print("Note: Interpretation of metrics")
    print("-"*80)
    print("1. PixCorr, SSIM, AlexNet(2), AlexNet(5), InceptionV3, CLIP: higher is better")
    print("2. EffNet-B, SwAV: distance metrics, lower is better")
    print("3. AlexNet/Inception/CLIP are 2-way identification accuracy (0-1)")
    print("4. 2-way identification: accuracy of matching reconstructed images to originals")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate reconstruction metrics using Jupyter Notebook methods')
    parser.add_argument('--base_path', type=str, default="/data2/ww/Uncertainty-aware-Blur-Prior-main",
                       help='base path')
    parser.add_argument('--image_paths_file', type=str, help='image path mapping file')
    
    args = parser.parse_args()
    
    if args.image_paths_file:
        image_paths_file = Path(args.image_paths_file)
    
    main()