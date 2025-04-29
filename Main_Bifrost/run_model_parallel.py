#!/usr/bin/env python
"""
Run inference using model parallelism (model sharded across GPUs) instead of data parallelism.
This is more memory-efficient for large models.
"""
import os
import sys
import torch
import numpy as np
import random
import time
import cv2
import einops

# Set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Import GPU utilities if available
try:
    sys.path.append('.')  # Add current directory to path
    from utils_gpu import verify_multi_gpu_setup, print_gpu_utilization, optimize_gpu_memory
    has_gpu_utils = True
    verify_multi_gpu_setup()
    print_gpu_utilization()
except ImportError:
    print("Warning: utils_gpu.py not found, some monitoring capabilities will be disabled")
    has_gpu_utils = False

# Import main modules
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from omegaconf import OmegaConf
from annotator.util import resize_image, HWC3
from DPT.run_monodepth_api import run, initialize_dpt_model
from segment_anything import SamPredictor, sam_model_registry
import torch.nn as nn

# Create a DataParallel wrapper that maintains attribute access
class DataParallelWithAttributes(nn.DataParallel):
    """DataParallel wrapper that preserves attribute access."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# Load SAM model
def load_sam_model():
    """Load SAM model optimized for memory usage."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM model on device: {device}")

    try:
        # Optimize memory settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Use last GPU for SAM
        if torch.cuda.device_count() > 1:
            sam_device = torch.device(f"cuda:{torch.cuda.device_count()-1}")
            print(f"Using device {sam_device} for SAM")
            sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
            sam = sam.to(sam_device).half()  # Use half precision
        else:
            sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
            sam = sam.to(device).half()  # Use half precision
            
        # Initialize predictor
        predictor = SamPredictor(sam)
        print("SAM model initialized successfully")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        return sam, predictor
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        raise

# Load the main model with memory optimization
def load_main_model():
    """Load the main model using per-layer device assignment for memory optimization."""
    print("Loading main model...")
    
    # Load configuration
    config = OmegaConf.load('./configs/inference.yaml')
    model_ckpt = config.pretrained_model
    model_config = config.config_file
    
    try:
        # Create model
        model = create_model(model_config).cpu()
        state_dict = load_state_dict(model_ckpt, location='cpu')
        
        # Get device count
        n_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpus}")
        
        if n_gpus > 1:
            # Manually distribute model layers across GPUs
            print(f"Distributing model across {n_gpus} GPUs")
            
            # Load state dict
            model.load_state_dict(state_dict)
            del state_dict  # Free memory
            
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            params_per_gpu = total_params / n_gpus
            print(f"Total parameters: {total_params:,}, Target per GPU: {params_per_gpu:,.0f}")
            
            # Assign layers to GPUs
            current_gpu = 0
            current_gpu_params = 0
            
            # Move model parameters to different GPUs
            for name, module in list(model.named_children()):
                # Calculate module parameters
                module_params = sum(p.numel() for p in module.parameters())
                
                # If this module would overload the current GPU, move to next GPU
                if current_gpu_params + module_params > params_per_gpu and current_gpu < n_gpus - 1:
                    current_gpu += 1
                    current_gpu_params = 0
                
                # Move module to the selected GPU
                target_device = f"cuda:{current_gpu}"
                print(f"Moving {name} ({module_params:,} params) to {target_device}")
                module.to(target_device)
                current_gpu_params += module_params
                
                # Clear cache periodically
                torch.cuda.empty_cache()
            
            print("Model distributed across GPUs")
            
            # Verify distribution
            gpu_param_counts = [0] * n_gpus
            for name, param in model.named_parameters():
                if param.device.type == 'cuda':
                    gpu_param_counts[param.device.index] += param.numel()
            
            for i, count in enumerate(gpu_param_counts):
                print(f"GPU {i}: {count:,} parameters ({count/total_params*100:.1f}%)")
        else:
            # Single GPU - try half precision
            print("Using single GPU with half precision")
            model.load_state_dict(state_dict)
            model = model.half().cuda()
        
        # Create sampler
        ddim_sampler = DDIMSampler(model)
        return model, ddim_sampler
    
    except Exception as e:
        print(f"Error loading main model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Get actual model (handling DataParallel case)
def get_actual_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

# Call model method (handling DataParallel case)
def call_model_method(model, method_name, *args, **kwargs):
    actual_model = get_actual_model(model)
    method = getattr(actual_model, method_name)
    return method(*args, **kwargs)

# Main function
def main():
    # Initialize models
    torch.cuda.empty_cache()
    sam, predictor = load_sam_model()
    
    # Load DPT model
    print("Loading DPT model...")
    dpt_model, transform = initialize_dpt_model(
        model_path='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt',
        custom_data_parallel=DataParallelWithAttributes
    )
    # Use GPU 2 for DPT if multiple GPUs available
    if torch.cuda.device_count() > 2:
        dpt_device = torch.device(f"cuda:2")
    else:
        dpt_device = torch.device("cuda:0")
    dpt_model = dpt_model.to(dpt_device)
    
    # Load main model
    model, ddim_sampler = load_main_model()
    
    # Memory optimization
    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    
    # Print GPU memory usage
    if has_gpu_utils:
        print_gpu_utilization()
    
    # Now we can set up the inference process...
    print("All models loaded successfully! Ready for inference.")
    
    # ... rest of inference code ...
    # This would typically include loading input images, running the models,
    # and saving results, similar to the original run_inference.py

if __name__ == "__main__":
    main() 