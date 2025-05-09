import gradio as gr
from gradio_bbox_annotator import BBoxAnnotator
from PIL import Image

example = BBoxAnnotator().example_value()

# def get_bbox_coordinates(bbox_data):
#     # Debug: Show the raw input data
#     return f"Raw input: {bbox_data}"

def get_normalized_bboxes(bbox_data):
    image_path, boxes = bbox_data
    # Load image to get dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, label = box
        x = x_min / img_width
        y = y_min / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        normalized_boxes.append([x, y, width, height, label])
    return normalized_boxes

def get_bbox_coordinates(bbox_data):
    from PIL import Image
    image_path, boxes = bbox_data
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    result = "Normalized bounding boxes:\n"
    for i, box in enumerate(boxes, 1):
        x_min, y_min, x_max, y_max, label = box
        x = x_min / img_width
        y = y_min / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Check if coordinates are within valid range
        if x + width > 1 or y + height > 1:
            return f"Error: Box {i} coordinates are out of range [0,1]"
            
        result += f"\nBox {i}: [x={x:.2f}, y={y:.2f}, width={width:.2f}, height={height:.2f}], label={label}\n"
    return result


demo = gr.Interface(
    get_bbox_coordinates,
    BBoxAnnotator(value=example, show_label=False),  # input is interactive
    gr.Textbox(label="Coordinates", lines=10),  # output is text
    examples=[[example]],  # examples are in the gallery format
)


if __name__ == "__main__":
    demo.launch()

# Get all bounding box coordinates
annotations = bbox_annotator.value.annotations
for annotation in annotations:
    left = annotation.left
    top = annotation.top 
    right = annotation.right
    bottom = annotation.bottom
    label = annotation.label  # Optional label if one was assigned
