#!/usr/bin/env python
"""
Half-precision (FP16) version of the inference script for memory efficiency.
"""
import os
import sys
import torch
import cv2
import einops
import numpy as np
import random
import time

# Set CUDA_VISIBLE_DEVICES to use all GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Add memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Import utilities
try:
    from utils_gpu import verify_multi_gpu_setup, print_gpu_utilization
    verify_multi_gpu_setup()
    print_gpu_utilization()
except ImportError:
    print("Warning: utils_gpu.py not found")

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

# Custom SamPredictor for half precision
class HalfPrecisionSamPredictor(SamPredictor):
    """
    Modified SamPredictor that ensures all inputs are in half precision
    """
    def predict(self, *args, **kwargs):
        # Convert point_coords and point_labels to half precision if they exist
        if 'point_coords' in kwargs and kwargs['point_coords'] is not None:
            kwargs['point_coords'] = torch.tensor(kwargs['point_coords'], 
                                                device=self.device,
                                                dtype=torch.float16)
        if 'point_labels' in kwargs and kwargs['point_labels'] is not None:
            kwargs['point_labels'] = torch.tensor(kwargs['point_labels'], 
                                                device=self.device,
                                                dtype=torch.float16)
        return super().predict(*args, **kwargs)
        
    def predict_torch(self, *args, **kwargs):
        # Convert inputs to half precision
        for key in kwargs:
            if isinstance(kwargs[key], torch.Tensor) and kwargs[key].dtype == torch.float32:
                kwargs[key] = kwargs[key].half()
        return super().predict_torch(*args, **kwargs)
        
    def set_image(self, image, image_format="RGB"):
        """Sets a new image for the predictor."""
        # Process the image normally
        result = super().set_image(image, image_format)
        
        # Convert features to half precision
        if hasattr(self, 'features') and self.features is not None:
            self.features = self.features.half()
            
        if hasattr(self, 'original_size') and self.original_size is not None:
            if isinstance(self.original_size, torch.Tensor) and self.original_size.dtype == torch.float32:
                self.original_size = self.original_size.half()
                
        if hasattr(self, 'input_size') and self.input_size is not None:
            if isinstance(self.input_size, torch.Tensor) and self.input_size.dtype == torch.float32:
                self.input_size = self.input_size.half()
                
        return result

# Create a DataParallel wrapper that preserves attribute access
class DataParallelWithAttributes(nn.DataParallel):
    """DataParallel wrapper that allows accessing model attributes even after wrapping."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# Initialize GPU settings
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Load models in half precision
def load_models_fp16():
    """Load all models in half precision for memory efficiency."""
    # Clear cache before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load SAM model on the last GPU
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            sam_device = f"cuda:{torch.cuda.device_count()-1}"
            print(f"Loading SAM on device {sam_device}")
        else:
            sam_device = device
            
        sam = sam_model_registry["vit_h"](
            checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth"
        )
        sam = sam.to(sam_device).half()  # Convert to half precision
        
        # Use our custom half precision-aware predictor
        predictor = HalfPrecisionSamPredictor(sam)
        print("SAM model loaded in half precision with custom HalfPrecisionSamPredictor")
        
    except Exception as e:
        print(f"Error loading SAM: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load DPT model on second-to-last GPU
    try:
        if torch.cuda.device_count() > 2:
            dpt_device = f"cuda:{torch.cuda.device_count()-2}"
        else:
            dpt_device = "cuda:0"
            
        print(f"Loading DPT on device {dpt_device}")
        dpt_model, transform = initialize_dpt_model(
            model_path='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt'
        )
        dpt_model = dpt_model.to(dpt_device).half()  # Convert to half precision
        print("DPT model loaded in half precision")
        
    except Exception as e:
        print(f"Error loading DPT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load main model on remaining GPUs
    try:
        config = OmegaConf.load('./configs/inference.yaml')
        model_ckpt = config.pretrained_model
        model_config = config.config_file
        
        print("Creating model...")
        model = create_model(model_config).cpu()
        print("Loading state dict...")
        state_dict = load_state_dict(model_ckpt, location='cpu')
        
        print("Loading state dict to model...")
        model.load_state_dict(state_dict)
        del state_dict  # Free memory
        
        # Convert to half precision before moving to GPU
        model = model.half()
        print("Converted model to half precision")
        
        # Use DataParallel with first N-2 GPUs
        if torch.cuda.device_count() > 2:
            devices = list(range(torch.cuda.device_count() - 2))
            print(f"Using GPUs {devices} for main model")
            model = DataParallelWithAttributes(model, device_ids=devices)
        
        model = model.cuda()
        print("Main model loaded in half precision")
        
        # Create sampler
        ddim_sampler = DDIMSampler(model)
        
    except Exception as e:
        print(f"Error loading main model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return sam, predictor, dpt_model, transform, model, ddim_sampler

# Get actual model behind any wrapper
def get_actual_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

# Call method on actual model
def call_model_method(model, method_name, *args, **kwargs):
    actual_model = get_actual_model(model)
    method = getattr(actual_model, method_name)
    return method(*args, **kwargs)

# Main code
if __name__ == "__main__":
    # Use half precision for everything
    print("Loading all models in half precision (FP16) for memory efficiency")
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Load models
    sam, predictor, dpt_model, transform, model, ddim_sampler = load_models_fp16()
    
    # Set up other configurations
    save_memory = True
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    
    print("All models loaded successfully in half precision!")
    
    # Now ready to run inference... 