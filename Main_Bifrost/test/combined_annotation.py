import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from gradio_image_annotation import image_annotator

def create_blank_image(width=500, height=400):
    """Create a blank white image."""
    return Image.new('RGB', (width, height), 'white')

def draw_points(image, points):
    """Draw points on the image."""
    # Create a new image if None
    if image is None:
        img = create_blank_image()
    else:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            img = Image.fromarray(image)
        else:
            img = image.copy()
    
    draw = ImageDraw.Draw(img)
    
    # Draw each point
    for point in points:
        x, y = point
        # Draw a small circle at each point
        draw.ellipse([x-5, y-5, x+5, y+5], fill='red')
    
    return img

def on_image_click(image, evt: gr.SelectData, points):
    """Handle click events on the image."""
    # Get click coordinates
    x, y = evt.index
    
    # Add new point to the list
    points.append((x, y))
    
    # Draw all points on the image
    result_image = draw_points(image, points)
    
    # Return the image and updated points list
    return result_image, points

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

def process_annotations(image, points, annotations):
    """Combine point and bounding box annotations into a single view."""
    if image is None:
        return None, "No image provided"
    
    # Convert image to PIL Image if it's not already
    if not isinstance(image, Image.Image):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    # Draw points
    img = draw_points(img, points)
    
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
with gr.Blocks(css="""
.annotation-panel {
    background: #181a20;
    border-radius: 10px;
    padding: 18px 12px 12px 12px;
    margin: 0 32px;
    border: 1px solid #23262f;
    min-width: 540px;
    max-width: 540px;
    min-height: 520px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.centered-row {justify-content: center; align-items: center; margin-top: 24px; margin-bottom: 12px;}
.centered-btn {width: 320px; margin: 0 auto;}
.centered-json {max-width: 900px; margin: 0 auto;}
""") as demo:
    gr.Markdown("# Image Annotation Tools")
    
    with gr.Row():
        # Point Drawing Section
        with gr.Column(scale=1, elem_classes=["annotation-panel"]):
            gr.Markdown("### Foreground image / Object image")
            gr.Markdown("Add points on the image to create a better mask.")
            
            # Store points in a state variable
            points = gr.State([])
            
            # Input image
            input_image = gr.Image(
                value=None,  # No default image
                type="pil",
                interactive=True,
                height=400,
                width=500
            )
            
            # Handle click events
            input_image.select(
                on_image_click,
                inputs=[input_image, points],
                outputs=[input_image, points]
            )
        
        # Bounding Box Section
        with gr.Column(scale=1, elem_classes=["annotation-panel"]):
            gr.Markdown("### background image")
            gr.Markdown("create a bounding box to place your object with in")
            annotator = image_annotator(
                label_list=["object"],
                label_colors=[(0, 255, 0)],
                height=400,
                width=500
            )
    
    # Process button and combined output
    with gr.Row(elem_classes=["centered-row"]):
        process_btn = gr.Button("Process", variant="primary", elem_classes=["centered-btn"])
    with gr.Row(elem_classes=["centered-row"]):
        combined_json = gr.JSON(label="Combined Coordinates", elem_classes=["centered-json"])
    
    # Handle process button click
    process_btn.click(
        process_annotations,
        inputs=[input_image, points, annotator],
        outputs=[input_image, combined_json]
    )

if __name__ == "__main__":
    demo.launch() 