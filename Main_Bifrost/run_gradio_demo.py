import cv2
import einops
import numpy as np
import torch
import random
import gradio as gr
import os
import albumentations as A
from PIL import Image
import torchvision.transforms as T
from datasets.data_utils import * 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from cldm.hack import disable_verbosity, enable_sliced_attention


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


# Get available GPU count
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Function to get the least busy GPU
def get_free_gpu():
    if num_gpus <= 1:
        return 0
        
    free_memory = []
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = torch.cuda.get_device_properties(i).total_memory - allocated
        free_memory.append(free)
    
    return free_memory.index(max(free_memory))

config = OmegaConf.load('./configs/demo.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
use_interactive_seg = config.config_file

# Create model on CPU first
model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cpu'))

# Move model to GPU(s)
if num_gpus > 1:
    # Use DataParallel for multiple GPUs
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    print(f"Using DataParallel across {num_gpus} GPUs")
else:
    # Use single GPU
    device_id = get_free_gpu()
    torch.cuda.set_device(device_id)
    model = model.cuda()
    print(f"Using single GPU: {device_id}")

ddim_sampler = DDIMSampler(model)

# Load the interactive segmentation model if needed
if use_interactive_seg:
    from iseg.coarse_mask_refine_util import BaselineModel
    model_path = './iseg/coarse_mask_refine.pth'
    iseg_model = BaselineModel().eval()
    weights = torch.load(model_path, map_location='cpu')['state_dict']
    iseg_model.load_state_dict(weights, strict=True)
    
    # Move iseg_model to a free GPU if possible
    if num_gpus > 0:
        device_id = get_free_gpu()
        iseg_model = iseg_model.cuda(device_id)
        print(f"Interactive segmentation model on GPU: {device_id}")


def process_image_mask(image_np, mask_np):
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    
    # Move tensors to the same device as iseg_model
    if torch.cuda.is_available():
        img = img.cuda(iseg_model.device if hasattr(iseg_model, 'device') else 0)
        mask = mask.cuda(iseg_model.device if hasattr(iseg_model, 'device') else 0)
    
    pred = iseg_model(img, mask)['instances'][0,0].detach().cpu().numpy() > 0.5 
    return pred.astype(np.uint8)

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
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

def inference_single_image(ref_image, 
                           ref_mask, 
                           tar_image, 
                           tar_mask, 
                           strength, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           ):
    raw_background = tar_image.copy()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control = enable_shape_control)

    ref = item['ref']
    hint = item['hint']
    num_samples = 1

    # Get the device where the model resides
    if isinstance(model, torch.nn.DataParallel):
        model_device = model.device_ids[0]
    else:
        model_device = next(model.parameters()).device

    # Move data to the appropriate device
    control = torch.from_numpy(hint.copy()).float().to(model_device)
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().to(model_device)
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H,W = 512,512

    # Set random seed if specified
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Get learned conditioning
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    cond = {"c_concat": [control], "c_crossattn": [module.get_learned_conditioning(clip_input)]}
    zero_input = torch.zeros((1, 3, 224, 224), device=model_device)
    un_cond = {"c_concat": [control], 
               "c_crossattn": [module.get_learned_conditioning([zero_input] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        if isinstance(model, torch.nn.DataParallel):
            model.module.low_vram_shift(is_diffusing=True)
        else:
            model.low_vram_shift(is_diffusing=True)

    # Set control scales
    if isinstance(model, torch.nn.DataParallel):
        model.module.control_scales = ([strength] * 13)
    else:
        model.control_scales = ([strength] * 13)
    
    # Sample
    samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                     shape, cond, verbose=False, eta=0,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=un_cond)

    if save_memory:
        if isinstance(model, torch.nn.DataParallel):
            model.module.low_vram_shift(is_diffusing=False)
        else:
            model.low_vram_shift(is_diffusing=False)

    # Decode samples
    if isinstance(model, torch.nn.DataParallel):
        x_samples = model.module.decode_first_stage(samples)
    else:
        x_samples = model.decode_first_stage(samples)
    
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

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
                tar_box_yyxx=np.array(tar_box_yyxx_full),
                 ) 
    return item


ref_dir='./examples/Gradio/FG'
image_dir='./examples/Gradio/BG'
ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
ref_list.sort()
image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
image_list.sort()

def mask_image(image, mask):
    blanc = np.ones_like(image) * 255
    mask = np.stack([mask,mask,mask],-1) / 255
    masked_image = mask * ( 0.5 * blanc + 0.5 * image) + (1-mask) * image
    return masked_image.astype(np.uint8)

def run_local(base,
              ref,
              *args):
    image = base["image"].convert("RGB")
    mask = base["mask"].convert("L")
    ref_image = ref["image"].convert("RGB")
    ref_mask = ref["mask"].convert("L")
    image = np.asarray(image)
    mask = np.asarray(mask)
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.')

    if mask.sum() == 0:
        raise gr.Error('No mask for the background image.')

    # Free up CUDA cache to maximize available memory
    torch.cuda.empty_cache()

    if reference_mask_refine:
        ref_mask = process_image_mask(ref_image, ref_mask)

    synthesis = inference_single_image(ref_image.copy(), ref_mask.copy(), image.copy(), mask.copy(), *args)
    synthesis = torch.from_numpy(synthesis).permute(2, 0, 1)
    synthesis = synthesis.permute(1, 2, 0).numpy()
    return [synthesis]

# Prepare GPU info display
gpu_info = ""
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    gpu_info += f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM\n"

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("#  Play with AnyDoor to Teleport your Target Objects! ")
        
        # Display GPU information
        if num_gpus > 0:
            gr.Markdown(f"### Running on {num_gpus} GPU(s):\n{gpu_info}")
            
        with gr.Row():
            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1, height=768)
            with gr.Accordion("Advanced Option", open=True):
                num_samples = 1
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=4.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1)
                reference_mask_refine = gr.Checkbox(label='Reference Mask Refine', value=False, interactive = True)
                enable_shape_control = gr.Checkbox(label='Enable Shape Control', value=False, interactive = True)
                
                gr.Markdown("### Guidelines")
                gr.Markdown(" Higher guidance-scale makes higher fidelity, while lower one makes more harmonized blending.")
                gr.Markdown(" Users should annotate the mask of the target object, too coarse mask would lead to bad generation.\
                              Reference Mask Refine provides a segmentation model to refine the coarse mask. ")
                gr.Markdown(" Enable shape control means the generation results would consider user-drawn masks to control the shape & pose; otherwise it \
                              considers the location and size to adjust automatically.")

    
        gr.Markdown("# Upload / Select Images for the Background (left) and Reference Object (right)")
        gr.Markdown("### You could draw coarse masks on the background to indicate the desired location and shape.")
        gr.Markdown("### <u>Do not forget</u> to annotate the target object on the reference image.")
        with gr.Row():
            base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512, brush_color='#FFFFFF', mask_opacity=0.5)
            ref = gr.Image(label="Reference", source="upload", tool="sketch", type="pil", height=512, brush_color='#FFFFFF', mask_opacity=0.5)
        run_local_button = gr.Button(label="Generate", value="Run")

        with gr.Row():
            with gr.Column():
                gr.Examples(image_list, inputs=[base],label="Examples - Background Image",examples_per_page=16)
            with gr.Column():
                gr.Examples(ref_list, inputs=[ref],label="Examples - Reference Object",examples_per_page=16)
        
    run_local_button.click(fn=run_local, 
                           inputs=[base, 
                                   ref, 
                                   strength, 
                                   ddim_steps, 
                                   scale, 
                                   seed,
                                   enable_shape_control, 
                                   ], 
                           outputs=[baseline_gallery]
                        )

# Add additional options for server launch
demo.launch(server_name="0.0.0.0")
