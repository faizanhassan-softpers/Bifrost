#!/bin/bash

# Script to run the model-parallel version of the inference code

# Explicitly set CUDA_VISIBLE_DEVICES to use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set additional optimization options
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Print configuration
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "PyTorch memory allocation config: $PYTORCH_CUDA_ALLOC_CONF"
echo "Running inference with model parallelism..."

# First clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run the inference script
python run_model_parallel.py 