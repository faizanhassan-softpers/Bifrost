import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np
from PIL import Image, ImageDraw
from gradio_image_annotation import image_annotator
import os
import tempfile
import uuid
from run_inference_lib import run_inference

def scale_points(prompts):
    image = prompts["image"]
    points = prompts["points"]
    
    if image is not None and points is not None:
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Scale points
        scaled_points = []
        for point in points:
            x = point[0] / width
            y = point[1] / height
            scaled_points.append([round(x, 3), round(y, 3)])
        
        return image, scaled_points
    return image, points

def get_boxes_json(annotations):
    boxes = annotations["boxes"]
    image_height, image_width = annotations["image"].shape[:2]
    
    normalized_boxes = []
    for box in boxes:
        # Convert to normalized coordinates [x, y, w, h] with 3 decimal places
        x = round(box["xmin"] / image_width, 3)
        y = round(box["ymin"] / image_height, 3)
        w = round((box["xmax"] - box["xmin"]) / image_width, 3)
        h = round((box["ymax"] - box["ymin"]) / image_height, 3)
        normalized_boxes.append([x, y, w, h])
    
    return normalized_boxes

def save_uploaded_image(image, prefix="image"):
    """Save uploaded image to a temporary file and return its path."""
    print(f"\n=== Debug: {prefix} Saving Process ===")
    print(f"Image is None: {image is None}")
    if image is not None:
        print(f"Image type: {type(image)}")
        print(f"Image shape/dimensions: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        
    if image is None:
        return None
        
    # Create directory if it doesn't exist
    save_dir = "./examples/TEST/Input"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate random filename
    random_filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(save_dir, random_filename)
    
    try:
        if isinstance(image, np.ndarray):
            print(f"Saving {prefix} numpy array image")
            Image.fromarray(image).save(save_path)
        elif isinstance(image, Image.Image):
            print(f"Saving {prefix} PIL Image")
            image.save(save_path)
        else:
            print(f"Unknown image type: {type(image)}")
            return None
            
        print(f"Image saved to: {save_path}")
        print(f"File exists: {os.path.exists(save_path)}")
        return save_path
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None
    print("===================================\n")

def process_annotations(bg_prompts, ref_prompts, points_df, annotations):
    """Combine point and bounding box annotations into a single view."""
    print("\n=== Debug: Checking conditions ===")
    print(f"bg_prompts is None: {bg_prompts is None}")
    print(f"ref_prompts is None: {ref_prompts is None}")
    
    if bg_prompts is not None:
        print(f"bg_prompts has image key: {'image' in bg_prompts}")
        if 'image' in bg_prompts:
            print(f"Background image type: {type(bg_prompts['image'])}")
    
    if ref_prompts is not None:
        print(f"ref_prompts has image key: {'image' in ref_prompts}")
        if 'image' in ref_prompts:
            print(f"Reference image type: {type(ref_prompts['image'])}")
    
    if bg_prompts is None or "image" not in bg_prompts:
        return None, "No background image provided"
    
    if ref_prompts is None or "image" not in ref_prompts:
        return None, "No reference image provided"
    
    bg_image = bg_prompts["image"]
    ref_image = ref_prompts["image"]
    
    if bg_image is None:
        return None, "No background image provided"
    
    if ref_image is None:
        return None, "No reference image provided"
    
    # Save both images and get their paths
    bg_image_path = save_uploaded_image(bg_image, prefix="bg")
    ref_image_path = save_uploaded_image(ref_image, prefix="ref")
    print(f"Returned bg_image_path: {bg_image_path}")
    print(f"Returned ref_image_path: {ref_image_path}")
    
    # Convert background image to PIL Image if it's not already
    if not isinstance(bg_image, Image.Image):
        img = Image.fromarray(bg_image)
    else:
        img = bg_image.copy()
    
    # Convert DataFrame to list of points
    points = []
    if not points_df.empty:
        points = points_df.values.tolist()
    
    # Draw points
    if points:
        draw = ImageDraw.Draw(img)
        for point in points:
            x, y = point
            draw.ellipse([x-5, y-5, x+5, y+5], fill='red')
    
    # Draw bounding boxes
    if annotations and "boxes" in annotations:
        draw = ImageDraw.Draw(img)
        for box in annotations["boxes"]:
            draw.rectangle(
                [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                outline="green",
                width=2
            )
    
    # Prepare combined output text
    output_text = {
        "points": points,
        "bounding_boxes": get_boxes_json(annotations) if annotations else []
    }
    
    # Call run_inference if we have both points and bounding boxes
    if points and annotations and "boxes" in annotations:
        # Get the first bounding box coordinates
        box = annotations["boxes"][0]
        bg_mask = [
            box["xmin"] / bg_image.shape[1],  # x
            box["ymin"] / bg_image.shape[0],  # y
            (box["xmax"] - box["xmin"]) / bg_image.shape[1],  # width
            (box["ymax"] - box["ymin"]) / bg_image.shape[0]   # height
        ]
        
        # Get all point coordinates normalized
        ref_object_location = []
        for point in points:
            x = point[0]
            y = point[1]
            ref_object_location.append([round(x, 6), round(y, 6)])
        
        # Run inference with static temp directory
        temp_dir_path = "./examples/temp"
        
        print("\n=== Values being passed to run_inference ===")
        print(f"temp_dir_path: {temp_dir_path}")
        print(f"bg_image_path: {bg_image_path}")
        print(f"ref_image_path: {ref_image_path}")
        print(f"bg_mask: {bg_mask}")
        print(f"ref_object_location: {ref_object_location}")
        print("===========================================\n")
        
        # Call the actual inference function
        gen_image, vis_image = run_inference(
            temp_dir_path=temp_dir_path,
            bg_image_path=bg_image_path,
            ref_image_path=ref_image_path,
            bg_mask=bg_mask,
            ref_object_location=ref_object_location
        )
        return img, output_text, gen_image, vis_image
        
        # For testing, return dummy images
        # dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # return img, output_text, dummy_image, dummy_image
    
    # Return dummy images when no inference is performed
    dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
    return img, output_text, dummy_image, dummy_image

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        # Background image input
        with gr.Column():
            gr.Markdown("### Background image")
            bg_annotator = image_annotator(
                label_list=["object"],
                label_colors=[(0, 255, 0)],
                height=400,
                width=500
            )
        
        # Reference image input
        with gr.Column():
            gr.Markdown("### Reference image")
            ref_annotator = ImagePrompter(show_label=False)
            point_output = gr.Dataframe(label="Points")
            
            ref_annotator.change(
                scale_points,
                inputs=[ref_annotator],
                outputs=[gr.Image(show_label=False), point_output]
            )
    
    # Process button and combined output
    process_btn = gr.Button("Process")
    combined_json = gr.JSON(label="Combined Coordinates")
    result_image = gr.Image(label="Generated Result")
    vis_image = gr.Image(label="Visualization")
    
    # Handle process button click
    process_btn.click(
        process_annotations,
        inputs=[bg_annotator, ref_annotator, point_output, bg_annotator],
        outputs=[gr.Image(show_label=False), combined_json, result_image, vis_image]
    )

if __name__ == "__main__":
    demo.launch(share=True) 