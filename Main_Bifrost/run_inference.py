import cv2
import einops
import numpy as np
import torch
import random
import os
import sys
import time

# Make sure this is at the top before any other imports that might use CUDA
# Set CUDA_VISIBLE_DEVICES to use specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Import GPU utilities
try:
    from utils_gpu import verify_multi_gpu_setup, print_gpu_utilization, print_pytorch_gpu_memory, optimize_gpu_memory
    has_gpu_utils = True
except ImportError:
    print("Warning: utils_gpu.py not found, some GPU monitoring capabilities will be disabled")
    has_gpu_utils = False

# sys.path.insert(0, '/home/ec2-user/SageMaker/Codes/ControlNet')

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from annotator.util import resize_image, HWC3
from datasets.data_utils import * 
from DPT.run_monodepth_api import run, initialize_dpt_model
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributed as dist

# Create a wrapper class that maintains access to model attributes when using DataParallel
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

# Initialize GPU settings
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Initialize models with GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SAM model on device: {device}")

try:
    # Optimize memory settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Load SAM model and move to device
    print("Loading SAM model...")
    
    # If multiple GPUs, use a specific device for SAM to leave memory for the main model
    if torch.cuda.device_count() > 1:
        # Use the last GPU for SAM to leave the first ones for the main model
        sam_device = torch.device(f"cuda:{torch.cuda.device_count()-1}")
        print(f"Using device {sam_device} for SAM to optimize memory usage")
        sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
        sam = sam.to(sam_device)
    else:
        sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
        sam = sam.to(device)
        
    # Try to use half precision to save memory
    try:
        sam = sam.half()  # Convert to half precision
        print("Converted SAM model to half precision")
    except Exception as e:
        print(f"Could not convert SAM to half precision: {e}")

    # Initialize the predictor with the base model
    print("Creating SamPredictor...")
    predictor = SamPredictor(sam)
    print("SamPredictor created successfully")

    # Check available GPUs for SAM
    print(f"CUDA device count before SAM DataParallel: {torch.cuda.device_count()}")

    # Now wrap with DataParallelWithAttributes for faster processing if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for SAM model's forward pass")
        try:
            # Use our custom wrapper to maintain attribute access
            print("Creating DataParallelWithAttributes wrapper for SAM...")
            # First verify the class exists
            print(f"DataParallelWithAttributes class exists: {DataParallelWithAttributes is not None}")
            sam = DataParallelWithAttributes(sam)
            print(f"SAM model is using DataParallel: {isinstance(sam, nn.DataParallel)}")
            # Verify we can still access attributes through the wrapper
            print(f"Can still access image_encoder: {hasattr(sam, 'image_encoder')}")
        except Exception as e:
            print(f"DataParallel wrapping failed: {e}, continuing with single-GPU SAM")
    else:
        print(f"Using single GPU for SAM, device: {next(sam.parameters()).device}")
        
    # Clear CUDA cache after SAM initialization
    torch.cuda.empty_cache()
    print("Cleared CUDA cache after SAM initialization")
    
except Exception as e:
    print(f"Error initializing SAM model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing without DataParallel for SAM...")

print(f"Loading DPT model...")
try:
    dpt_model, transform = initialize_dpt_model(
        model_path='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt',
        custom_data_parallel=DataParallelWithAttributes
    )
    dpt_model = dpt_model.to(device)
    print("DPT model loaded successfully")
except Exception as e:
    print(f"Error initializing DPT model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing with limited functionality...")

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = config.pretrained_model
model_config = config.config_file

# Create model on CPU first
try:
    print("Creating model...")
    model = create_model(model_config).cpu()
    print("Loading state dict...")
    state_dict = load_state_dict(model_ckpt, location='cpu')  # Load to CPU first

    # Check if multiple GPUs are available
    print(f"CUDA device count before model init: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference!")
        
        # Load state dict on CPU first
        print("Loading state dict to model...")
        model.load_state_dict(state_dict)
        
        # Free up CPU memory
        del state_dict
        torch.cuda.empty_cache()
        
        # Create DataParallelWithAttributes wrapper
        print("Creating DataParallelWithAttributes wrapper for main model...")
        model = DataParallelWithAttributes(model)
        
        # Move to CUDA device by device to avoid OOM
        print("Moving model to CUDA gradually...")
        # First move model parts to different GPUs
        if hasattr(model.module, 'split_for_gpus'):
            print("Using model's built-in split_for_gpus method")
            model.module.split_for_gpus(torch.cuda.device_count())
        else:
            print("Using PyTorch's native DataParallel to distribute model")
            # Let PyTorch handle distribution
        
        # Move model to CUDA with memory monitoring
        try:
            for i, (name, param) in enumerate(model.named_parameters()):
                target_device = i % torch.cuda.device_count()
                param_size_mb = param.numel() * param.element_size() / (1024 * 1024)
                print(f"Moving parameter {name} ({param_size_mb:.2f} MB) to cuda:{target_device}")
                param.data = param.data.to(f"cuda:{target_device}")
                
                # Clear cache periodically
                if i % 20 == 0:
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM error during parameter distribution: {e}")
            print("Trying alternative approach with smaller chunks...")
            
            # Reset model to CPU
            model = model.cpu()
            torch.cuda.empty_cache()
            
            # Try again with model sharding
            print("Creating model with device_map='auto' for automatic sharding...")
            # We'll use the first GPU as the entry point
            torch.cuda.set_device(0)
            model = model.cuda()
        
        print(f"Model is using DataParallel: {isinstance(model, nn.DataParallel)}")
        print(f"Model device: {next(model.parameters()).device}")
    else:
        # Single GPU case
        print("Loading state dict...")
        model.load_state_dict(state_dict)
        print("Moving model to CUDA...")
        try:
            model = model.cuda()
        except torch.cuda.OutOfMemoryError:
            print("Out of memory on single GPU. Trying to load with lower precision...")
            # Try with half-precision
            model = model.half().cuda()
        
        print(f"Using single GPU, device: {next(model.parameters()).device}")

    # Create sampler
    print("Creating DDIMSampler...")
    ddim_sampler = DDIMSampler(model)
    print("DDIMSampler created successfully")
except Exception as e:
    print(f"Error initializing main model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)  # Exit if the main model fails to load

# Function to get the actual model (handles DataParallel case)
def get_actual_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

# Function to handle model method calls (handles DataParallel case)
def call_model_method(model, method_name, *args, **kwargs):
    actual_model = get_actual_model(model)
    method = getattr(actual_model, method_name)
    return method(*args, **kwargs)

def aug_tar_mask(mask, kernal_size=0.001):
    w, h = mask.shape[1], mask.shape[0]
    aug_mask = mask.copy()
    for i in range(h):
        for j in range(w):
            if mask[i,j] == 1:
                aug_mask[max(i-int(kernal_size*h), 0):min(i+int(kernal_size*h), h),max(j-int(kernal_size*w),0):min(j+int(kernal_size*w),w)] = 1
    return aug_mask

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold):
    """Process image pairs for inference. Optimized for multi-GPU processing."""
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]
    
    # Getting the depth map
    tar_depth_norm = np.zeros(tar_depth.shape, dtype=np.float32)
    cv2.normalize(tar_depth, tar_depth_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tar_depth_norm = tar_depth_norm.astype(np.uint8)
    tar_depth_norm = HWC3(tar_depth_norm)

    # Augmenting reference image
    # masked_ref_image = aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
 
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255, color=sobel_color, thresh=sobel_threshold)
    
    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    cropped_tar_depth = tar_depth_norm[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    occluded_mask = cv2.resize(occluded_mask.astype(np.uint8), (x2-x1, y2-y1))
    
    for i in range(ref_image_collage.shape[0]):
        for j in range(ref_image_collage.shape[1]):
            if occluded_mask[i,j] == 0:
                ref_image_collage[i, j] = 0.0
                
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0
    
    if pixel_num != None:
        cropped_tar_mask = aug_tar_mask(cropped_tar_mask, pixel_num)
    collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)
    for i in range(cropped_tar_mask.shape[0]):
        for j in range(cropped_tar_mask.shape[1]):
            if cropped_tar_mask[i,j] == 0:
                collage[i, j] = cropped_target_image[i,j]
            elif i<y1 or i>=y2 or j<x1 or j>=x2:
                if cropped_tar_mask[i,j] == 1:
                    collage[i, j] = 0.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)
    
    tar_depth_norm = pad_to_square(cropped_tar_depth, pad_value = 0, random = False)
    tar_depth = cv2.resize(tar_depth_norm.astype(np.uint8), (512,512)).astype(np.uint8)
    H, W, C = tar_depth.shape
    tar_depth = cv2.resize(tar_depth, (W, H), interpolation=cv2.INTER_LINEAR)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]], -1)
    tar_depth = tar_depth / 255

    item = dict(ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                depth=tar_depth.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop)) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


# Add a function to manage GPU memory
def manage_gpu_memory():
    """Clean up GPU memory and report usage statistics."""
    if torch.cuda.is_available():
        # Report memory usage
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        # Empty cache to free up memory
        torch.cuda.empty_cache()
        print("GPU cache cleared")

# Verify GPU availability right after imports
print("\nVerifying GPU availability:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
# Force PyTorch to re-detect GPUs if only one is found
if torch.cuda.device_count() < 2 and ',' in os.environ.get("CUDA_VISIBLE_DEVICES", "0"):
    print("Warning: Expected multiple GPUs but only found one. Attempting to fix...")
    # Clear CUDA cache
    torch.cuda.empty_cache()
    # Re-initialize CUDA
    if hasattr(torch.cuda, 'reset_max_memory_allocated'):
        torch.cuda.reset_max_memory_allocated()
    if hasattr(torch.cuda, 'reset_max_memory_cached'):
        torch.cuda.reset_max_memory_cached()
    # Print updated count
    print(f"Updated GPU count: {torch.cuda.device_count()}")

# Update inference_single_image function to include memory management
def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num=10, sobel_color=False, sobel_threshold=20, guidance_scale = 5.0):
    # Starting memory check
    if torch.cuda.is_available():
        print("Memory status at start of inference:")
        manage_gpu_memory()
        
    # Process the image pairs
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold)
    
    # Prepare visualization
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5
    depth = item['depth'] * 255
    num_samples = 1
    hint_image = hint[:,:,:-1]
    hint_mask = hint[:,:,-1]
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))
    vis = cv2.hconcat([ref.astype(np.float32), depth.astype(np.float32), hint_image.astype(np.float32), hint_mask.astype(np.float32)])
    cv2.imwrite('sample_vis_test.jpg',vis[:,:,::-1])
    
    seed = random.randint(0, 65535)
    if save_memory:
        call_model_method(model, "low_vram_shift", is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    depth = item['depth']
    
    # Check memory before moving tensors to GPU
    if torch.cuda.is_available():
        print("Memory status before loading tensors to GPU:")
        manage_gpu_memory()
    
    # Move tensors to GPU - replicate for all GPUs if using DataParallel
    control_detail = torch.from_numpy(hint.copy()).float().cuda() 
    control_detail = torch.stack([control_detail for _ in range(num_samples)], dim=0)
    control_detail = einops.rearrange(control_detail, 'b h w c -> b c h w').clone()
    
    control_depth = torch.from_numpy(depth.copy()).float().cuda() 
    control_depth = torch.stack([control_depth for _ in range(num_samples)], dim=0)
    control_depth = einops.rearrange(control_depth, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H, W = 512, 512

    # Get the actual model for conditioning
    actual_model = get_actual_model(model)
    
    # Check memory after tensor preparation
    if torch.cuda.is_available():
        print("Memory status after tensor preparation:")
        manage_gpu_memory()
    
    # Prepare conditions for the model
    learned_conditioning = actual_model.get_learned_conditioning(clip_input)
    
    # Set up conditional and unconditional inputs
    cond = {
        "c_concat_detail": [control_detail], 
        "c_concat_depth": [control_depth], 
        "c_crossattn": [learned_conditioning]
    }
    
    un_cond = {
        "c_concat_detail": None if guess_mode else [control_detail], 
        "c_concat_depth": None if guess_mode else [control_depth], 
        "c_crossattn": [actual_model.get_learned_conditioning([torch.zeros((1,3,224,224), device=clip_input.device)] * num_samples)]
    }

    shape = (4, H // 8, W // 8)

    if save_memory:
        call_model_method(model, "low_vram_shift", is_diffusing=True)

    # Sampling parameters
    num_samples = 1
    image_resolution = 512
    strength = 1
    guess_mode = False
    ddim_steps = 50
    scale = guidance_scale
    seed = -1
    eta = 0.0

    # Set control scales with the actual model
    actual_model = get_actual_model(model)
    actual_model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

    # Check memory before sampling
    if torch.cuda.is_available():
        print("Memory status before sampling:")
        manage_gpu_memory()
    
    # Sample with DataParallel model
    samples, intermediates = ddim_sampler.sample(
        ddim_steps, 
        num_samples,
        shape, 
        cond, 
        verbose=False, 
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )
    
    if save_memory:
        call_model_method(model, "low_vram_shift", is_diffusing=False)

    # Check memory after sampling
    if torch.cuda.is_available():
        print("Memory status after sampling:")
        manage_gpu_memory()
    
    # Decode samples with the actual model
    actual_model = get_actual_model(model)
    x_samples = actual_model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    
    # Final memory cleanup
    if torch.cuda.is_available():
        print("Final memory cleanup:")
        manage_gpu_memory()
        
    return gen_image


def depth_mask_fusion(back_depth, ref_depth, back_mask, ref_mask, depth_scale=[0, 0.5], mode='place'):
    w, h = back_depth.shape[1], back_depth.shape[0]
    tar_mask = np.ones(back_depth.shape[:2], np.uint8)
    tar_mask[int((back_mask[1])*h):int(back_mask[1]*h)+int(back_mask[3]*h),
            int((back_mask[0])*w):int(back_mask[0]*w)+int(back_mask[2]*w)] = 0
    normalized_back_depth = np.zeros(back_depth.shape, dtype=np.float32)
    cv2.normalize(back_depth, normalized_back_depth, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_ref_depth = np.zeros(ref_depth.shape, dtype=np.float32)
    cv2.normalize(ref_depth, normalized_ref_depth, alpha=depth_scale[0]*255, beta=depth_scale[1]*255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    back_depth = normalized_back_depth.copy()
    ref_depth = normalized_ref_depth.copy()
    scaled_ref_depth = cv2.resize(ref_depth, (int(w*back_mask[2]), int(h*back_mask[3])))
    scaled_ref_mask = cv2.resize(ref_mask, (int(w*back_mask[2]), int(h*back_mask[3])))
    selected_area = back_depth[int(h*back_mask[1]):int(h*back_mask[1])+scaled_ref_depth.shape[0], int(w
           * back_mask[0]):int(w*(back_mask[0])+scaled_ref_depth.shape[1])]
    selected_mask = scaled_ref_mask
    if mode == 'place':
        for i in range(selected_area.shape[0]):
            for j in range(selected_area.shape[1]):
                if selected_mask[i,j] == 1 and scaled_ref_depth[i, j] > selected_area[i, j]:
                    selected_area[i, j] = scaled_ref_depth[i, j]
                else:
                    selected_mask[i, j] = 0
    elif mode == 'replace':
        for i in range(selected_area.shape[0]):
            selected_area[i, :] = np.linspace(selected_area[i,0], selected_area[i,-1], selected_area.shape[1])
        for i in range(selected_area.shape[0]):
            for j in range(selected_area.shape[1]):
                '''
                if selected_mask[i, j] == 0:
                    selected_mask[i, j] = 1
                else:
                    selected_mask[i, j] = 0
                    '''
                if selected_mask == 0 and scaled_ref_depth[i, j]>selected_area[i,j]:
                    selected_area[i, j] = scaled_ref_depth[i, j]
                else:
                    selected_area[i, j] = 0
                    selected_mask[i, j] = 0
    elif mode == 'draw':
        for i in range(selected_area.shape[0]):
            for j in range(selected_area.shape[1]):
                continue
                    
    back_depth[int(h*back_mask[1]):int(h*back_mask[1])+scaled_ref_depth.shape[0], int(w
           * back_mask[0]):int(w*(back_mask[0])+scaled_ref_depth.shape[1])] = selected_area
    tar_mask[int(h*back_mask[1]):int(h*back_mask[1])+scaled_ref_depth.shape[0], int(w
           * back_mask[0]):int(w*(back_mask[0])+scaled_ref_depth.shape[1])] = 1-selected_mask
    
    return back_depth, 1-tar_mask, selected_mask


if __name__ == '__main__': 
    # Set the CUDA device and verify all GPUs are detected
    if torch.cuda.is_available():
        print(f"\nFinal GPU verification before starting inference:")
        
        # Use our new utilities if available
        if has_gpu_utils:
            verify_multi_gpu_setup()
            print_gpu_utilization()
            print_pytorch_gpu_memory()
        else:
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
            
            # Print GPU information 
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # If we only have 1 GPU but should have more, try resetting CUDA
        if torch.cuda.device_count() < 2 and ',' in os.environ.get("CUDA_VISIBLE_DEVICES", "0"):
            print("Still only seeing 1 GPU. Trying alternative approach...")
            # Try forcing device visibility a different way
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')
            print(f"Should have {len(visible_devices)} GPUs: {visible_devices}")
            
            # Try updating the environment and reloading CUDA
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Use our memory optimization utility if available
            if has_gpu_utils:
                optimize_gpu_memory()
            else:
                torch.cuda.empty_cache()
            
            # Print what we have
            print(f"Devices after reset:")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # When using DataParallel, the primary device is the first one
        torch.cuda.set_device(0)
        
        # Initial memory status
        print("Initial GPU memory status:")
        if has_gpu_utils:
            print_pytorch_gpu_memory()
        else:
            manage_gpu_memory()
    
    # ==== Example for inferring a single image ===
    ref_image_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/object.jpg'
    ref_image_mask_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Mask/object_mask.jpg'
    ref_image_depth_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Depth/object.png'

    bg_image_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/background.jpg'
    bg_mask_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Mask/background_mask.png'
    bg_image_depth_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/Test/Depth/background.png'

    fused_depth_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Depth/fused_depth.png'
    fused_mask_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Mask/fused_mask.png'

    save_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Gen/gen_res.png'
    save_compose_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Gen/gen_res_compose.png'

    input_folder = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input'
    output_folder = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Depth'
    # [x, y, w, h] in the range of [0, 1]
    bg_mask = [0.338, 0.521, 0.2, 0.32]
    ref_object_location = [0.5, 0.45] # [x, y] in the range of [0, 1]
    bg_object_location = [0.6, 0.5] # [x, y] in the range of [0, 1]
    depth = [0.1, 0.22] # the range of scaled depth value
    pixel_num = 0.02 # the number of pixels added around the mask for augmentation default 10
    mode = 'place' # 'place', 'replace', 'draw
    flip_image = False
    sobel_color = False
    sobel_threshold = 50
    start_time = time.time()

    print("Loading reference and background images...")
    # reference image
    image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if flip_image:
        image = cv2.flip(image, 1)

    h, w = image.shape[0], image.shape[1]

    # background image
    back_image = cv2.imread(bg_image_path).astype(np.uint8)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # Memory check after loading images
    if torch.cuda.is_available():
        print("Memory status after loading images:")
        manage_gpu_memory()

    print("Using SAM to predict the mask for reference image...")
    # Use SAM to predict the mask for reference image
    predictor.set_image(image)
    point_coords = np.array([[h*ref_object_location[1], w*ref_object_location[0]]])
    point_labels = np.array([1])
    masks, _, _ = predictor.predict(point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True)
    # save the mask image
    mask = masks[1].astype(np.uint8)
    # cv2.imwrite(ref_image_mask_path, mask)
    mask = cv2.imread(ref_image_mask_path, cv2.IMREAD_UNCHANGED)
    ref_mask = (mask[:, :] > 0).astype(np.uint8)
    ref_image = image
    if mode == 'draw':
        h_back, w_back = back_image.shape[0], back_image.shape[1]
        # Use SAM to predict the mask for background image
        predictor.set_image(back_image)
        point_coords = np.array([[w_back*bg_object_location[0], h_back*bg_object_location[1]]])
        point_labels = np.array([1])
        masks, _, _ = predictor.predict(point_coords=point_coords,
                                    point_labels=point_labels,
                                    multimask_output=True)
        # save the mask image
        back_mask = masks[1].astype(np.uint8)
        # cv2.imwrite(bg_mask_path, back_mask)

    # Memory check after mask prediction
    if torch.cuda.is_available():
        print("Memory status after mask prediction:")
        manage_gpu_memory()

    print("Getting depth maps using DPT...")
    # Get the depth map using DPT
    run(dpt_model, transform, input_folder, output_folder)

    # Memory check after depth prediction
    if torch.cuda.is_available():
        print("Memory status after depth prediction:")
        manage_gpu_memory()

    tar_mask = np.zeros(back_image.shape[:2], np.uint8)
    tar_mask[int((bg_mask[1])*back_image.shape[0]):int((bg_mask[1]+bg_mask[3])*back_image.shape[0]),
            int((bg_mask[0])*back_image.shape[1]):int((bg_mask[0]+bg_mask[2])*back_image.shape[1])] = 1

    print("Processing depth maps...")
    # read the depth map predicted by DPT
    back_depth = cv2.imread('/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Depth/background.png', cv2.IMREAD_UNCHANGED)
    ref_depth = cv2.imread('/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Depth/object.png', cv2.IMREAD_UNCHANGED)
    ref_depth = ref_depth*ref_mask
    if flip_image:
        ref_depth = cv2.flip(ref_depth, 1)

    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    y1,y2,x1,x2 = ref_box_yyxx
    ref_depth = ref_depth[y1:y2,x1:x2]
    cropped_ref_mask = ref_mask[y1:y2,x1:x2]

    print("Fusing depth and mask...")
    # fuse the depth and mask
    fused_depth, fused_mask, occluded_mask = depth_mask_fusion(back_depth, ref_depth, bg_mask, cropped_ref_mask, depth, mode=mode)
    fused_mask = fused_mask*255
    cv2.imwrite(fused_mask_path, fused_mask)
    cv2.imwrite(fused_depth_path, fused_depth)

    # background mask
    if mode == 'place' and pixel_num != None: 
        tar_mask = cv2.imread(fused_mask_path)[:,:,0] > 128
        tar_mask = tar_mask.astype(np.uint8)
    elif mode == 'draw':
        tar_mask = cv2.imread(bg_mask_path, cv2.IMREAD_UNCHANGED)
        tar_mask = (tar_mask[:, :] > 0).astype(np.uint8)
    
    if flip_image:
        ref_mask = cv2.flip(ref_mask, 1)

    tar_depth = cv2.imread(fused_depth_path, cv2.IMREAD_UNCHANGED)
    
    # Memory check before inference
    if torch.cuda.is_available():
        print("Memory status before running main inference:")
        manage_gpu_memory()
    
    print("Running main inference...")
    # Run inference using the model distributed across multiple GPUs
    gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold)
    
    # Memory check after inference
    if torch.cuda.is_available():
        print("Memory status after inference:")
        manage_gpu_memory()
    
    print("Saving results...")
    h,w = back_image.shape[0], back_image.shape[0]
    ref_image = cv2.resize(ref_image, (w,h))
    tar_depth = cv2.resize(tar_depth, (w,h))
    vis_image = cv2.hconcat([ref_image, back_image, gen_image])

    cv2.imwrite(save_compose_path, vis_image[:,:,::-1])
    cv2.imwrite(save_path, gen_image[:,:,::-1])
    end_time = time.time()
    print("Total time: ", end_time-start_time)
    
    # Final memory cleanup
    if torch.cuda.is_available():
        print("Final memory cleanup:")
        manage_gpu_memory()
        
    print("Inference completed successfully!")

    

