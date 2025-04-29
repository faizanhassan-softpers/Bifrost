"""Compute depth maps for images in the input folder.
"""
import sys
sys.path.insert(
    0, '/home/ec2-user/dev/Bifrost/Main_Bifrost/DPT')
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.midas_net import MidasNet_large
from dpt.models import DPTDepthModel
from torchvision.transforms import Compose
import util.io
import os
import glob
import torch
import torch.nn as nn
import cv2
import argparse
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from DPT.util.misc import visualize_attention
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Allow DPT to use all available GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# from util.misc import visualize_attention


def run(model, transform, input_path, output_path, model_type="dpt_large"):
    """Run the DPT model on all images in a directory."""
    # Get device from model (handles both normal and DataParallel models)
    if isinstance(model, nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device
    
    print(f"Running DPT on device: {device}")
    
    # Get all images in the directory
    if os.path.isdir(input_path):
        img_names = glob.glob(os.path.join(input_path, "*.jpg"))
        img_names += glob.glob(os.path.join(input_path, "*.jpeg"))
        img_names += glob.glob(os.path.join(input_path, "*.png"))
    else:
        img_names = [input_path]

    os.makedirs(output_path, exist_ok=True)

    for ind, img_name in enumerate(img_names):
        img = cv2.imread(img_name)
        if img is None:
            print(f"Error: Unable to load image {img_name}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(img, 3, axis=2)

        # Prepare image for model input
        img_input = transform({"image": img})["image"]
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)
        
        # Perform inference with memory optimization
        with torch.no_grad():
            prediction = model.forward(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        prediction = prediction.cpu().numpy()
        
        # Save output
        output_name = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0] + ".png"
        )
        cv2.imwrite(output_name, prediction)

    print(f"Finished processing {len(img_names)} images. Results saved in {output_path}")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize_dpt_model(model_path, model_type="dpt_large", optimize=True, custom_data_parallel=None):
    """Initialize the DPT model with multi-GPU support.
    
    Args:
        model_path: Path to the model weights
        model_type: Type of DPT model to load
        optimize: Whether to optimize memory usage
        custom_data_parallel: Custom DataParallel class to use (if None, uses standard DataParallel)
    """
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    
    # Load model
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        print(f"Model {model_type} not implemented, please choose one of the other models.")
        return None, None

    if optimize:
        # use PyTorch to perform memory optimizations
        model.to(memory_format=torch.channels_last)
    
    # Move model to GPU
    model = model.to(device)
    model.eval()
    
    # Check if multiple GPUs are available and use DataParallel if so
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DPT model")
        # Use custom DataParallel class if provided
        if custom_data_parallel is not None:
            model = custom_data_parallel(model)
        else:
            model = torch.nn.DataParallel(model)
    
    # load transforms
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model, transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_large",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth",
                        dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Initialize model
    model, transform = initialize_dpt_model(
        args.model_weights, args.model_type, args.optimize)

    # compute depth maps
    run(
        model,
        transform,
        args.input_path,
        args.output_path,
        args.model_type,
    )
