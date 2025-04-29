#!/bin/bash

# Script to run the half-precision (FP16) version of the inference code
# This is highly optimized for memory usage

# Explicitly set CUDA_VISIBLE_DEVICES to use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set optimization options
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Print configuration
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "PyTorch memory configuration: $PYTORCH_CUDA_ALLOC_CONF"
echo "Running inference in half-precision (FP16) mode..."

# Clear GPU memory first
python -c "import torch; torch.cuda.empty_cache()"

# Run the half-precision inference script
python run_inference_fp16.py 