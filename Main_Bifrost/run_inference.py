import cv2
import einops
import numpy as np
import torch
import random
import os
import sys
import time
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
dpt_model, transform = initialize_dpt_model(model_path='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt')


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



def aug_tar_mask(mask, kernal_size=0.001):
    w, h = mask.shape[1], mask.shape[0]
    aug_mask = mask.copy()
    for i in range(h):
        for j in range(w):
            if mask[i,j] == 1:
                aug_mask[max(i-int(kernal_size*h), 0):min(i+int(kernal_size*h), h),max(j-int(kernal_size*w),0):min(j+int(kernal_size*w),w)] = 1
    return aug_mask



def process_pairs(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold):
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
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    '''
    occluded_mask_resized = cv2.resize(occluded_mask.astype(np.uint8), (masked_ref_image_compose.shape[1], masked_ref_image_compose.shape[0]))
    for i in range(masked_ref_image_compose.shape[0]):
        for j in range(masked_ref_image_compose.shape[1]):
            if occluded_mask_resized[i,j] == 0:
                masked_ref_image_compose[i, j] = 0.0
                '''  
    # ref_box_yyxx = get_bbox_from_mask(ref_mask_compose)
    # y1,y2,x1,x2 = ref_box_yyxx
    # masked_ref_image_compose = masked_ref_image_compose[y1:y2,x1:x2,:]
    # ref_mask_compose = ref_mask_compose[y1:y2,x1:x2]
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
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
    tar_depth = tar_depth / 255

    item = dict(ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                depth=tar_depth.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
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


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num=10, sobel_color=False, sobel_threshold=20, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold)
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
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    depth = item['depth']
    
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
    H,W = 512,512

    cond = {"c_concat_detail": [control_detail], "c_concat_depth": [control_depth], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat_detail": None if guess_mode else [control_detail], "c_concat_depth": None if guess_mode else [control_depth], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
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


def process_images_at_runtime(object_image_path, background_image_path, 
                             output_dir="./output", 
                             sam_model_path="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth",
                             dpt_model_path='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt',
                             ref_object_location=[0.5, 0.45],
                             bg_object_location=[0.6, 0.5],
                             bg_mask=[0.338, 0.521, 0.2, 0.32],
                             depth_scale=[0.1, 0.22],
                             pixel_num=0.02,
                             mode='place',
                             flip_image=False,
                             sobel_color=False,
                             sobel_threshold=50,
                             guidance_scale=5.0):
    """
    Process images provided at runtime and generate all intermediary images.
    
    Args:
        object_image_path: Path to the object image
        background_image_path: Path to the background image
        output_dir: Directory to save all output images
        sam_model_path: Path to the SAM model weights
        dpt_model_path: Path to the DPT model weights
        ref_object_location: [x, y] coordinates of the object in reference image
        bg_object_location: [x, y] coordinates of where to place object in background
        bg_mask: [x, y, w, h] of the mask area in background
        depth_scale: Range of scaled depth values
        pixel_num: Number of pixels for mask augmentation
        mode: 'place', 'replace', or 'draw'
        flip_image: Whether to flip the input image
        sobel_color: Whether to use color in sobel edge detection
        sobel_threshold: Threshold for sobel edge detection
        guidance_scale: Scale for guidance in the diffusion model
    
    Returns:
        Path to the generated image
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(output_dir, "Input")
    mask_dir = os.path.join(output_dir, "Mask")
    depth_dir = os.path.join(output_dir, "Depth")
    gen_dir = os.path.join(output_dir, "Gen")
    
    for dir_path in [input_dir, mask_dir, depth_dir, gen_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy input images to the input directory
    object_filename = os.path.basename(object_image_path)
    bg_filename = os.path.basename(background_image_path)
    
    object_copy_path = os.path.join(input_dir, object_filename)
    bg_copy_path = os.path.join(input_dir, bg_filename)
    
    # Copy/save input images
    cv2.imwrite(object_copy_path, cv2.imread(object_image_path))
    cv2.imwrite(bg_copy_path, cv2.imread(background_image_path))
    
    # Define paths for intermediate files
    ref_image_mask_path = os.path.join(mask_dir, f"{os.path.splitext(object_filename)[0]}_mask.jpg")
    bg_mask_path = os.path.join(mask_dir, f"{os.path.splitext(bg_filename)[0]}_mask.png")
    ref_image_mask_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Mask/object_mask.jpg'
    # ref_image_depth_path = os.path.join(depth_dir, f"{os.path.splitext(object_filename)[0]}.png")
    bg_image_depth_path = os.path.join(depth_dir, f"{os.path.splitext(bg_filename)[0]}.png")
    fused_depth_path = os.path.join(depth_dir, "fused_depth.png")
    fused_mask_path = os.path.join(mask_dir, "fused_mask.png")
    save_path = os.path.join(gen_dir, "gen_res.png")
    save_compose_path = os.path.join(gen_dir, "gen_res_compose.png")
    
    # Initialize SAM model
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
    predictor = SamPredictor(sam)
    
    # Initialize DPT model
    dpt_model, transform = initialize_dpt_model(model_path=dpt_model_path)
    
    start_time = time.time()
    
    # Validate mask boundaries
    if bg_mask[0]+bg_mask[2] > 1 or bg_mask[1]+bg_mask[3] > 1:
        print('The mask is out of range')
        return None
    
    # Process reference (object) image
    image = cv2.imread(object_image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if flip_image:
        image = cv2.flip(image, 1)
    
    h, w = image.shape[0], image.shape[1]
    
    # Process background image
    back_image = cv2.imread(background_image_path).astype(np.uint8)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
    
    # Use SAM to predict mask for reference image
    predictor.set_image(image)
    point_coords = np.array([[h*ref_object_location[1], w*ref_object_location[0]]])
    point_labels = np.array([1])
    masks, _, _ = predictor.predict(point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True)
    
    # Save the mask image
    mask = masks[1].astype(np.uint8) * 255
    cv2.imwrite(ref_image_mask_path, mask)
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
        # Save the mask image
        back_mask = masks[1].astype(np.uint8) * 255
        cv2.imwrite(bg_mask_path, back_mask)
    
    # Generate depth maps using DPT
    run(dpt_model, transform, input_dir, depth_dir)
    
    # Create target mask
    tar_mask = np.zeros(back_image.shape[:2], np.uint8)
    tar_mask[int((bg_mask[1])*back_image.shape[0]):int((bg_mask[1]+bg_mask[3])*back_image.shape[0]),
            int((bg_mask[0])*back_image.shape[1]):int((bg_mask[0]+bg_mask[2])*back_image.shape[1])] = 1
    
    # Read the depth maps predicted by DPT
    back_depth = cv2.imread(bg_image_depth_path, cv2.IMREAD_UNCHANGED)
    ref_depth = cv2.imread(ref_image_depth_path, cv2.IMREAD_UNCHANGED)
    ref_depth = ref_depth*ref_mask
    if flip_image:
        ref_depth = cv2.flip(ref_depth, 1)
    
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    
    # Filter ref mask
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    y1,y2,x1,x2 = ref_box_yyxx
    ref_depth = ref_depth[y1:y2,x1:x2]
    cropped_ref_mask = ref_mask[y1:y2,x1:x2]
    
    # Fuse the depth and mask
    fused_depth, fused_mask, occluded_mask = depth_mask_fusion(back_depth, ref_depth, bg_mask, cropped_ref_mask, depth_scale, mode=mode)
    fused_mask = fused_mask*255
    cv2.imwrite(fused_mask_path, fused_mask)
    cv2.imwrite(fused_depth_path, fused_depth)
    
    # Handle background mask
    if mode == 'place' and pixel_num != None: 
        tar_mask = cv2.imread(fused_mask_path)[:,:,0] > 128
        tar_mask = tar_mask.astype(np.uint8)
    elif mode == 'draw':
        tar_mask = cv2.imread(bg_mask_path, cv2.IMREAD_UNCHANGED)
        tar_mask = (tar_mask[:, :] > 0).astype(np.uint8)
    
    if flip_image:
        ref_mask = cv2.flip(ref_mask, 1)
    
    tar_depth = cv2.imread(fused_depth_path, cv2.IMREAD_UNCHANGED)
    
    # Generate the final image
    gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold, guidance_scale)
    
    # Create visualization
    h,w = back_image.shape[0], back_image.shape[0]
    ref_image = cv2.resize(ref_image, (w,h))
    tar_depth = cv2.resize(tar_depth, (w,h))
    vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    # Save output images
    cv2.imwrite(save_compose_path, vis_image[:,:,::-1])
    cv2.imwrite(save_path, gen_image[:,:,::-1])
    
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)
    
    # Return the path to the generated image
    return save_path

if __name__ == '__main__': 
    # Parse command line arguments for runtime use
    parser = argparse.ArgumentParser(description='Bifrost Image Generation')
    parser.add_argument('--object_image', type=str, required=False,
                        help='Path to the object image')
    parser.add_argument('--background_image', type=str, required=False,
                        help='Path to the background image')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output images')
    parser.add_argument('--sam_model_path', type=str, 
                        default="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth",
                        help='Path to the SAM model weights')
    parser.add_argument('--dpt_model_path', type=str,
                        default='/home/ec2-user/SageMaker/model_weights/dpt_large-midas-2f21e586.pt',
                        help='Path to the DPT model weights')
    parser.add_argument('--ref_object_location', type=float, nargs=2, default=[0.5, 0.45],
                        help='[x, y] coordinates of the object in reference image')
    parser.add_argument('--bg_object_location', type=float, nargs=2, default=[0.6, 0.5],
                        help='[x, y] coordinates of where to place object in background')
    parser.add_argument('--bg_mask', type=float, nargs=4, default=[0.338, 0.521, 0.2, 0.32],
                        help='[x, y, w, h] of the mask area in background')
    parser.add_argument('--depth_scale', type=float, nargs=2, default=[0.1, 0.22],
                        help='Range of scaled depth values')
    parser.add_argument('--pixel_num', type=float, default=0.02,
                        help='Number of pixels for mask augmentation')
    parser.add_argument('--mode', type=str, choices=['place', 'replace', 'draw'], default='place',
                        help='Mode for object placement')
    parser.add_argument('--flip_image', action='store_true',
                        help='Whether to flip the input image')
    parser.add_argument('--sobel_color', action='store_true',
                        help='Whether to use color in sobel edge detection')
    parser.add_argument('--sobel_threshold', type=int, default=50,
                        help='Threshold for sobel edge detection')
    parser.add_argument('--guidance_scale', type=float, default=5.0,
                        help='Scale for guidance in the diffusion model')
    
    args = parser.parse_args()
    
    # If object_image and background_image are provided via command line, use them
    if args.object_image and args.background_image:
        result_path = process_images_at_runtime(
            args.object_image,
            args.background_image,
            args.output_dir,
            args.sam_model_path,
            args.dpt_model_path,
            args.ref_object_location,
            args.bg_object_location,
            args.bg_mask,
            args.depth_scale,
            args.pixel_num,
            args.mode,
            args.flip_image,
            args.sobel_color,
            args.sobel_threshold,
            args.guidance_scale
        )
        print(f"Generated image saved to: {result_path}")
    else:
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

        # using zero123 to generate new image with novel view
        # given_image = cv2.imread(opt.input_image, cv2.IMREAD_UNCHANGED)
        # given_image = cv2.cvtColor(given_image.copy(), cv2.COLOR_BGR2RGB)
        # run_demo(opts=opt)

        # reference image + reference mask
        # You could use the demo of SAM to extract RGB-A image with masks
        # https://segment-anything.com/demo

        if bg_mask[0]+bg_mask[2] > 1 or bg_mask[1]+bg_mask[3] > 1:
            print('The mask is out of range')
            exit()

        # reference image
        image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        if flip_image:
            image = cv2.flip(image, 1)

        h, w = image.shape[0], image.shape[1]



        # background image
        back_image = cv2.imread(bg_image_path).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
        '''
        image_gry = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

        # mask = (image[:,:,-1] > 128).astype(np.uint8)
        mask = (image_gry[:, :] < 253).astype(np.uint8)
        # image = image[:,:,:-1]
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        ref_image = image*mask[:, :, None]
        ref_image = image 
        ref_mask = mask
        '''
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

        # Get the depth map using DPT
        run(dpt_model, transform, input_folder, output_folder)


        # transform reference image style to the background image style
        # aug = A.Compose([A.FDA([back_image], p=1, read_fn=lambda x: x)])
        # transfered_ref_image = aug(image=image)['image']
        # ref_image = transfered_ref_image


        tar_mask = np.zeros(back_image.shape[:2], np.uint8)
        tar_mask[int((bg_mask[1])*back_image.shape[0]):int((bg_mask[1]+bg_mask[3])*back_image.shape[0]),
                    int((bg_mask[0])*back_image.shape[1]):int((bg_mask[0]+bg_mask[2])*back_image.shape[1])] = 1


        # read the depth map predicted by DPT
        back_depth = cv2.imread('./examples/TEST/Depth/background.png', cv2.IMREAD_UNCHANGED)
        ref_depth = cv2.imread('./examples/TEST/Depth/object.png', cv2.IMREAD_UNCHANGED)
        ref_depth = ref_depth*ref_mask
        if flip_image:
            ref_depth = cv2.flip(ref_depth, 1)

        ref_box_yyxx = get_bbox_from_mask(ref_mask)

        # ref filter mask 
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        y1,y2,x1,x2 = ref_box_yyxx
        ref_depth = ref_depth[y1:y2,x1:x2]
        cropped_ref_mask = ref_mask[y1:y2,x1:x2]

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
        # tar_mask = tar_mask < 128
        # tar_mask = tar_mask.astype(np.uint8)
        if flip_image:
            ref_mask = cv2.flip(ref_mask, 1)

        tar_depth = cv2.imread(fused_depth_path, cv2.IMREAD_UNCHANGED)
        gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask, occluded_mask, tar_depth, pixel_num, sobel_color, sobel_threshold)
        # print("gen_image: ", gen_image.shape)
        h,w = back_image.shape[0], back_image.shape[0]
        ref_image = cv2.resize(ref_image, (w,h))
        tar_depth = cv2.resize(tar_depth, (w,h))
        # given_image = cv2.resize(given_image, (w,h))
        vis_image = cv2.hconcat([ref_image, back_image, gen_image])

        cv2.imwrite(save_compose_path, vis_image[:,:,::-1])
        cv2.imwrite(save_path, gen_image[:,:,::-1])
        end_time = time.time()
        print("Time: ", end_time-start_time)

    '''
    # ==== Example for inferring VITON-HD Test dataset ===

    from omegaconf import OmegaConf
    import os 
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = '/home/mhf/dxl/Lingxiao/Codes/dreambooth/dreambooth_output_anydoor'
    flag = False
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    image_names = os.listdir(test_dir)
    object_dir = '/home/mhf/dxl/Lingxiao/Codes/dreambooth/dreambooth_object'
    bg_dir = '/home/mhf/dxl/Lingxiao/Codes/dreambooth/bg_test'
    # output_dir = '/mnt/workspace/gongkaixiong/lingxiaoli/Codes/dreambooth/dreambooth_output_anydoor'
    image_names = os.listdir(object_dir)
    for image_name in image_names:
        if image_name.endswith('.jpg'):
            ref_image_path = os.path.join(object_dir, image_name)
            print("ref_image_path: ", ref_image_path)
            background_names = os.listdir(bg_dir)
            for background_name in tqdm(background_names):
                if not background_name.endswith("mask.jpg"):
                    print("background_name: ", background_name)
                    background_path = os.path.join(bg_dir, background_name)
                    count = background_name.split('_')[-1].split('.')[0]
                    # background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
                    # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                    # end_string = background_name.split('.')[0].split('_')[-1]
                    # backgroun_mask = cv2.imread(background_path.replace('.jpg', '_mask.jpg'), cv2.IMREAD_UNCHANGED)
                    
                    # ref_image_path = os.path.join(test_dir, image_name)
                    tar_image_path = background_path
                    ref_mask_path = ref_image_path.replace('/dreambooth_object/','/object_mask/')
                    tar_mask_path = tar_image_path.replace('.jpg', '_mask.jpg')

                    ref_image = cv2.imread(ref_image_path)
                    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

                    gt_image = cv2.imread(tar_image_path)
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

                    ref_mask = (cv2.imread(ref_mask_path) > 0).astype(np.uint8)[:,:,0]

                    tar_mask = Image.open(tar_mask_path ).convert('P')
                    tar_mask= np.array(tar_mask)
                    tar_mask = (tar_mask < 1).astype(np.uint8)

                    gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
                    dir_name = image_name.split('.')[0]
                    if not os.path.exists(os.path.join(save_dir, dir_name)):
                        os.mkdir(os.path.join(save_dir, dir_name))
                    gen_path = os.path.join(save_dir, dir_name, str(count)+'.jpg')
                    # vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
                    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(gen_path, gen_image)
                    # count += 1
                    flag = True
                # if flag:
                #     break
            # if flag:
            #     break    
    '''

    

