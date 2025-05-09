import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

def create_blank_image(width=800, height=600):
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
    
    # Format coordinates as a list of lists
    coords_text = str(points)
    
    # Return the image, coordinates, and updated points list
    return result_image, coords_text, points

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Point Drawing")
    gr.Markdown("Click on the image to draw points and see their coordinates.")
    
    # Store points in a state variable
    points = gr.State([])
    
    with gr.Row():
        # Input image
        input_image = gr.Image(
            value=create_blank_image(),
            type="pil",
            interactive=True,
            height=600,
            width=800
        )
        
        # Output coordinates
        coordinates = gr.Textbox(
            label="Coordinates",
            value="[]",
            lines=10  # Make the textbox taller to show more coordinates
        )
    
    # Handle click events
    input_image.select(
        on_image_click,
        inputs=[input_image, points],
        outputs=[input_image, coordinates, points]
    )

if __name__ == "__main__":
    demo.launch() 