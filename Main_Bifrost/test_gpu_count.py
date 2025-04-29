#!/usr/bin/env python
"""
Simple script to test GPU detection and count.
"""
import os
import torch
import sys

# Set CUDA_VISIBLE_DEVICES explicitly at the start
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Print details for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    # Create a small tensor on each GPU to test
    print("\nTesting tensor creation on each GPU:")
    for i in range(torch.cuda.device_count()):
        # Set the current GPU
        torch.cuda.set_device(i)
        # Create a tensor on this GPU
        x = torch.ones(1, device=f'cuda:{i}')
        print(f"Tensor on GPU {i}, device: {x.device}")
        
    # Test DataParallel
    if torch.cuda.device_count() > 1:
        print("\nTesting DataParallel:")
        # Simple model for testing
        model = torch.nn.Linear(10, 10)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        print(f"Model is on device: {next(model.parameters()).device}")
        print(f"Model is using DataParallel: {isinstance(model, torch.nn.DataParallel)}")
else:
    print("CUDA is not available on this system") 