import cv2
import einops
import numpy as np
import torch
import random
import os
import argparse
from PIL import Image
import torchvision.transforms as T
from datasets.data_utils import * 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from cldm.hack import disable_verbosity, enable_sliced_attention

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Use all available GPUs
gpu_ids = list(range(num_gpus))
gpu_ids_str = ','.join(map(str, gpu_ids))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
print(f"Using GPUs: {gpu_ids_str}")

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Enable memory saving for all operations
save_memory = True
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configure device allocation based on available GPUs
if num_gpus >= 2:
    # Multi-GPU setup
    # Interactive segmentation is smallest, put on GPU 0
    device_iseg = torch.device("cuda:0")
    # Main model is largest, put on remaining GPU with most memory
    device_model = torch.device("cuda:1")
else:
    # Only 1 GPU - need to use more aggressive memory management
    device_iseg = torch.device("cuda:0")
    device_model = torch.device("cuda:0")

print(f"Interactive segmentation on {device_iseg}, Main model on {device_model}")

# Enable memory tracking for debugging
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/(1024**2):.2f}MB / {torch.cuda.get_device_properties(i).total_memory/(1024**2):.2f}MB")

# Recursively move an entire model to the device
def move_to_device(model, device):
    if hasattr(model, 'to'):
        model = model.to(device)
    
    # Recursively move internal models
    for attr_name in dir(model):
        try:
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.nn.Module):
                setattr(model, attr_name, move_to_device(attr, device))
            elif attr_name == 'model' and hasattr(attr, 'to'):
                setattr(model, attr_name, attr.to(device))
        except: 
            pass
    return model

def process_image_mask(image_np, mask_np, iseg_model):
    # Ensure we're using the interactive segmentation model's GPU
    torch.cuda.set_device(device_iseg.index)
    
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0).to(device_iseg)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device_iseg)
    
    with torch.no_grad():
        pred = iseg_model(img, mask)['instances'][0,0].detach().cpu().numpy() > 0.5 
    
    return pred.astype(np.uint8)

def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 3 # maigin_pixel

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
    tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return tar_image

def inference_single_image(model, ddim_sampler, ref_image, ref_mask, tar_image, tar_mask, 
                           strength, ddim_steps, scale, seed, enable_shape_control, model_type=None):
    # Ensure we're using the main model's GPU
    torch.cuda.set_device(device_model.index)
    
    # Set random seed if provided
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    raw_background = tar_image.copy()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control=enable_shape_control)

    ref = item['ref']
    hint = item['hint']
    num_samples = 1

    # Move tensors to the correct device
    control = torch.from_numpy(hint.copy()).float().to(device_model)
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().to(device_model)
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H,W = 512,512

    # Create empty tensor on the same device for conditioning
    empty_tensor = torch.zeros((1,3,224,224), device=device_model)
    
    # Check if model type is provided through args
    is_bifrost_model = False
    if model_type == "bifrost":
        is_bifrost_model = True
        print("Using Bifrost model type specified by user")
    elif model_type == "standard":
        is_bifrost_model = False
        print("Using standard model type specified by user")
    else:
        # Auto-detect model type
        try:
            # Check model config file name as a hint
            config_path = os.path.basename(model_config).lower() if 'model_config' in locals() else ""
            if 'bifrost' in config_path:
                is_bifrost_model = True
                print(f"Detected Bifrost model based on config filename: {config_path}")
        except Exception as e:
            print(f"Warning when checking model type: {e}")
            print("Using standard conditioning structure")
    
    # Create a depth tensor (all zeros) if needed for Bifrost model
    depth_tensor = None
    if is_bifrost_model:
        print("Creating depth tensor for Bifrost model")
        depth_tensor = torch.zeros((num_samples, 3, H, W), device=device_model)
    
    # Set up conditioning based on detected model type
    if is_bifrost_model:
        print("Using Bifrost conditioning structure with detail and depth")
        cond = {
            "c_concat_detail": [control],
            "c_concat_depth": [depth_tensor], 
            "c_crossattn": [model.get_learned_conditioning(clip_input)]
        }
        un_cond = {
            "c_concat_detail": [control],
            "c_concat_depth": [depth_tensor],
            "c_crossattn": [model.get_learned_conditioning(empty_tensor)]
        }
    else:
        # For standard ControlNet models
        print("Using standard ControlNet conditioning structure")
        cond = {
            "c_concat": [control], 
            "c_crossattn": [model.get_learned_conditioning(clip_input)]
        }
        un_cond = {
            "c_concat": [control], 
            "c_crossattn": [model.get_learned_conditioning(empty_tensor)]
        }
    
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = ([strength] * 13)
    
    # Use mixed precision to save memory
    with torch.cuda.amp.autocast(enabled=save_memory):
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                       shape, cond, verbose=False, eta=0,
                                       unconditional_guidance_scale=scale,
                                       unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    tar_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 

    # keep background unchanged
    y1,y2,x1,x2 = item['tar_box_yyxx']
    raw_background[y1:y2, x1:x2, :] = tar_image[y1:y2, x1:x2, :]
    return raw_background


def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8, enable_shape_control = False):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]

    ratio = np.random.randint(11, 15) / 10 #11,13
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # collage aug 
    masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    tar_box_yyxx_full = tar_box_yyxx
    
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]

    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0
    if enable_shape_control:
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    masked_ref_image = masked_ref_image  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
    
    item = dict(ref=masked_ref_image.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ),
                tar_box_yyxx=np.array(tar_box_yyxx_full)) 
    return item


def main():
    parser = argparse.ArgumentParser(description="AnyDoor CLI - Teleport objects between images")
    
    # Required parameters
    parser.add_argument("--background", type=str, required=True, help="Path to background image")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image containing object")
    parser.add_argument("--output", type=str, required=True, help="Path to save output image")
    
    # Optional parameters with static directories for masks
    parser.add_argument("--background-mask", type=str, help="Path to background mask (if not provided, will use static directory)")
    parser.add_argument("--reference-mask", type=str, help="Path to reference object mask (if not provided, will use static directory)")
    parser.add_argument("--mask-dir", type=str, default="./masks", help="Static directory to store masks")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength (0.0-2.0)")
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps")
    parser.add_argument("--scale", type=float, default=4.5, help="Guidance scale (0.1-30.0)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--shape-control", action="store_true", help="Enable shape control")
    parser.add_argument("--refine-mask", action="store_true", help="Enable reference mask refinement")
    parser.add_argument("--config", type=str, default="./configs/demo.yaml", help="Path to config file")
    parser.add_argument("--model-type", type=str, choices=["bifrost", "standard"], help="Force model type (bifrost or standard)")
    
    args = parser.parse_args()
    
    # Create static mask directories if they don't exist
    os.makedirs(args.mask_dir, exist_ok=True)
    bg_mask_dir = os.path.join(args.mask_dir, "background")
    ref_mask_dir = os.path.join(args.mask_dir, "reference")
    os.makedirs(bg_mask_dir, exist_ok=True)
    os.makedirs(ref_mask_dir, exist_ok=True)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    model_ckpt = config.pretrained_model
    model_config = config.config_file
    use_interactive_seg = config.config_file

    # Load iseg model if needed
    iseg_model = None
    if args.refine_mask and use_interactive_seg:
        torch.cuda.set_device(device_iseg.index)
        torch.cuda.empty_cache()
        from iseg.coarse_mask_refine_util import BaselineModel
        model_path = './iseg/coarse_mask_refine.pth'
        iseg_model = BaselineModel().eval()
        weights = torch.load(model_path, map_location='cpu')['state_dict']
        iseg_model.load_state_dict(weights, strict=True)
        iseg_model = iseg_model.to(device_iseg)
        print(f"Interactive segmentation model loaded for mask refinement")

    # Create and load main model
    torch.cuda.set_device(device_model.index)
    torch.cuda.empty_cache()
    print(f"Loading main model from {model_ckpt}")
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cpu'))
    model = model.to(device_model)
    ddim_sampler = DDIMSampler(model)
    print(f"Model loaded successfully")

    # Load images
    print("Loading background image...")
    background = cv2.imread(args.background)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    print("Loading reference image...")
    reference = cv2.imread(args.reference)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    
    # Determine mask paths
    bg_filename = os.path.basename(args.background)
    ref_filename = os.path.basename(args.reference)
    bg_mask_path = args.background_mask if args.background_mask else os.path.join(bg_mask_dir, f"{os.path.splitext(bg_filename)[0]}_mask.png")
    ref_mask_path = args.reference_mask if args.reference_mask else os.path.join(ref_mask_dir, f"{os.path.splitext(ref_filename)[0]}_mask.png")
    
    # Check if masks exist, otherwise create default masks
    if not os.path.exists(bg_mask_path):
        print(f"Background mask not found at {bg_mask_path}, creating a default mask...")
        # Create a simple mask in the center (1/4 of the image)
        h, w = background.shape[:2]
        background_mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h//2, w//2
        size_y, size_x = h//4, w//4
        background_mask[cy-size_y//2:cy+size_y//2, cx-size_x//2:cx+size_x//2] = 255
        cv2.imwrite(bg_mask_path, background_mask)
        print(f"Default background mask created at {bg_mask_path}")
    else:
        print(f"Loading background mask from {bg_mask_path}")
        background_mask = cv2.imread(bg_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if not os.path.exists(ref_mask_path):
        print(f"Reference mask not found at {ref_mask_path}, creating a default mask...")
        # Create a mask that covers most of the reference image
        h, w = reference.shape[:2]
        reference_mask = np.zeros((h, w), dtype=np.uint8)
        margin = min(h, w) // 8
        reference_mask[margin:h-margin, margin:w-margin] = 255
        cv2.imwrite(ref_mask_path, reference_mask)
        print(f"Default reference mask created at {ref_mask_path}")
    else:
        print(f"Loading reference mask from {ref_mask_path}")
        reference_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert masks to binary
    background_mask = (background_mask > 128).astype(np.uint8)
    reference_mask = (reference_mask > 128).astype(np.uint8)
    
    # Validate masks
    if reference_mask.sum() == 0:
        print("Warning: Reference mask is empty. Creating basic central mask.")
        h, w = reference.shape[:2]
        reference_mask = np.zeros((h, w), dtype=np.uint8)
        margin = min(h, w) // 4
        reference_mask[margin:h-margin, margin:w-margin] = 1
        cv2.imwrite(ref_mask_path, reference_mask * 255)
    
    if background_mask.sum() == 0:
        print("Warning: Background mask is empty. Creating basic central mask.")
        h, w = background.shape[:2]
        background_mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h//2, w//2
        size_y, size_x = h//4, w//4
        background_mask[cy-size_y//2:cy+size_y//2, cx-size_x//2:cx+size_x//2] = 1
        cv2.imwrite(bg_mask_path, background_mask * 255)
    
    # Apply mask refinement if requested
    if args.refine_mask and iseg_model is not None:
        print("Refining reference mask...")
        reference_mask = process_image_mask(reference, reference_mask, iseg_model)
        # Save the refined mask
        cv2.imwrite(ref_mask_path, reference_mask * 255)
        print(f"Refined reference mask saved to {ref_mask_path}")
    
    # Run inference
    print(f"Starting inference with: strength={args.strength}, steps={args.steps}, scale={args.scale}, seed={args.seed}")
    result = inference_single_image(
        model, 
        ddim_sampler,
        reference, 
        reference_mask, 
        background, 
        background_mask, 
        args.strength, 
        args.steps, 
        args.scale, 
        args.seed,
        args.shape_control,
        args.model_type
    )
    
    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    cv2.imwrite(args.output, result_bgr)
    print(f"Result saved to {args.output}")
    
    # Clean up
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 