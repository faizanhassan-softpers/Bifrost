import gradio as gr
from gradio_image_annotation import image_annotator


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


with gr.Blocks() as demo:
    with gr.Tab("Object annotation", id="tab_object_annotation"):
        annotator = image_annotator(
            label_list=["object"],
            label_colors=[(0, 255, 0)],
        )
        button_get = gr.Button("Get bounding boxes")
        json_boxes = gr.JSON()
        button_get.click(get_boxes_json, annotator, json_boxes)

if __name__ == "__main__":
    demo.launch()
