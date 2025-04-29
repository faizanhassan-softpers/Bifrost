#!/bin/bash

# Explicitly set CUDA_VISIBLE_DEVICES to use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print configuration
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Running inference with multi-GPU support..."

# Run the inference script
python run_inference.py 