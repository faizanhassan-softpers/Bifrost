#!/usr/bin/env python
"""
Test script to verify SAM model initialization and DataParallel wrapping.
"""
import os
import sys
import torch
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry

# Set CUDA_VISIBLE_DEVICES to use all GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Import utility functions if available
try:
    from utils_gpu import verify_multi_gpu_setup, print_gpu_utilization
    verify_multi_gpu_setup()
    print_gpu_utilization()
except ImportError:
    print("utils_gpu.py not found, continuing without GPU utilities")

# Define our DataParallel wrapper
class DataParallelWithAttributes(nn.DataParallel):
    """
    DataParallel wrapper that allows accessing model attributes even after wrapping.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If attribute not found in DataParallel, check the module
            return getattr(self.module, name)

# Initialize models with GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SAM model on device: {device}")

try:
    # Load SAM model and move to device
    print("Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
    print("SAM model loaded")
    
    # Check if image_encoder is accessible before moving to device
    print(f"Original model has image_encoder: {hasattr(sam, 'image_encoder')}")
    print(f"image_encoder properties: {sam.image_encoder.__class__.__name__ if hasattr(sam, 'image_encoder') else 'N/A'}")
    
    # Move to device
    sam = sam.to(device)
    print(f"SAM model moved to device: {next(sam.parameters()).device}")
    
    # Initialize the predictor with the base model
    print("Creating SamPredictor...")
    predictor = SamPredictor(sam)
    print("SamPredictor created successfully")
    
    # Check GPU count
    print(f"CUDA device count before DataParallel: {torch.cuda.device_count()}")
    
    # Try accessing the model's attributes
    print("\nAccessing model attributes before DataParallel:")
    for attribute in ['image_encoder', 'prompt_encoder', 'mask_decoder']:
        has_attr = hasattr(sam, attribute)
        print(f"- has {attribute}: {has_attr}")
        
    # Now wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs for SAM model")
        try:
            # First test with standard DataParallel
            print("\nTesting with standard DataParallel:")
            sam_dp = nn.DataParallel(sam)
            print("Standard DataParallel wrapping successful")
            
            # Try accessing attributes through standard DataParallel
            print("Accessing attributes through standard DataParallel:")
            try:
                print(f"- Direct access to image_encoder: {hasattr(sam_dp, 'image_encoder')}")
            except Exception as e:
                print(f"- Error accessing image_encoder: {type(e).__name__}: {e}")
                
            # Access through module
            print(f"- Access through module: {hasattr(sam_dp.module, 'image_encoder')}")
            
            # Now test with our custom wrapper
            print("\nTesting with custom DataParallelWithAttributes:")
            sam_custom = DataParallelWithAttributes(sam)
            print("Custom DataParallel wrapping successful")
            
            # Try accessing attributes through custom DataParallel
            print("Accessing attributes through custom DataParallel:")
            try:
                has_attr = hasattr(sam_custom, 'image_encoder')
                print(f"- Direct access to image_encoder: {has_attr}")
                if has_attr:
                    print(f"- image_encoder class: {sam_custom.image_encoder.__class__.__name__}")
            except Exception as e:
                print(f"- Error accessing image_encoder: {type(e).__name__}: {e}")
                
        except Exception as e:
            print(f"Error during DataParallel wrapping: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Only one GPU available, skipping DataParallel tests")
        
    print("\nSAM initialization tests completed successfully")
    
except Exception as e:
    print(f"Error during SAM initialization: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 