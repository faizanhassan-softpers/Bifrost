#!/bin/bash

# Explicitly set CUDA_VISIBLE_DEVICES to use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print configuration
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Verifying GPU setup..."

# First run the utility script to verify GPU setup
python -c "from utils_gpu import verify_multi_gpu_setup; verify_multi_gpu_setup()"

# If that succeeds, run the main script
echo "Running inference with multi-GPU support..."
python run_inference.py 