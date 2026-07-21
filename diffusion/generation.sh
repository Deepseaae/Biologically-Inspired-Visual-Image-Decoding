#!/bin/bash

# Script to run eeg_to_image_diffusion.py in training mode

# Path to the training EEG feature file (PyTorch tensor)
TRAIN_EEG_FEATURES="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/exp/eeg_intra-subject_baseline_EEGProjectLayer_RN50/sub-08_seed0/features/train_eeg_features.pt"

# Path to the training image feature file (PyTorch tensor)
TRAIN_IMAGE_FEATURES="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/exp/eeg_intra-subject_baseline_EEGProjectLayer_RN50/sub-08_seed0/features/train_img_features.pt"

# Directory where output results (generated images, logs, etc.) will be saved
OUTPUT_DIR="./eeg_to_image_results"

# Device to use for computation (e.g., "cuda" for GPU, "cpu" for CPU)
DEVICE="cuda"

# Path to the SDXL Turbo model weights (Stable Diffusion XL Turbo)
SDXL_PATH="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/models--stabilityai--sdxl-turbo"

# Path to the IP-Adapter model weights (for image prompt adaptation)
IP_ADAPTER_PATH="/data2/ww/Biologically-Inspired-Visual-Image-Decoding/data/ModelWeights/IP-Adapter"

# Run the Python script with all arguments
python eeg_to_image_diffusion.py \
    --train \
    --train_eeg_features "$TRAIN_EEG_FEATURES" \
    --train_image_features "$TRAIN_IMAGE_FEATURES" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --sdxl_path "$SDXL_PATH" \
    --ip_adapter_path "$IP_ADAPTER_PATH"