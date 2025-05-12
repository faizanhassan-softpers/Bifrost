import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np
from PIL import Image, ImageDraw
from gradio_image_annotation import image_annotator

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

def process_annotations(prompts, points_df, annotations):
    """Combine point and bounding box annotations into a single view."""
    if prompts is None or "image" not in prompts:
        return None, "No image provided"
    
    image = prompts["image"]
    if image is None:
        return None, "No image provided"
    
    # Convert image to PIL Image if it's not already
    if not isinstance(image, Image.Image):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
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
    
    return img, output_text

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        # Foreground image input
        with gr.Column():
            gr.Markdown("### Foreground image / Object image")
            point_annotator = ImagePrompter(show_label=False)
            point_output = gr.Dataframe(label="Points")
            
            point_annotator.change(
                scale_points,
                inputs=[point_annotator],
                outputs=[gr.Image(show_label=False), point_output]
            )
        
        # Background image input
        with gr.Column():
            gr.Markdown("### Background image")
            annotator = image_annotator(
                label_list=["object"],
                label_colors=[(0, 255, 0)],
                height=400,
                width=500
            )
    
    # Process button and combined output
    process_btn = gr.Button("Process")
    combined_json = gr.JSON(label="Combined Coordinates")
    
    # Handle process button click
    process_btn.click(
        process_annotations,
        inputs=[point_annotator, point_output, annotator],
        outputs=[gr.Image(show_label=False), combined_json]
    )

if __name__ == "__main__":
    demo.launch(share=True) 