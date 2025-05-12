import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os

def initialize_sam():
    """Initialize SAM model and predictor"""
    sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    return predictor

def predict_mask(image_path, point_coords, point_labels, output_path=None):
    """
    Predict mask using SAM for a single image
    
    Args:
        image_path (str): Path to input image
        point_coords (list): List of [x, y] coordinates for points
        point_labels (list): List of labels (1 for foreground, 0 for background)
        output_path (str, optional): Path to save the mask. If None, mask won't be saved
    
    Returns:
        numpy.ndarray: Predicted mask
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize SAM predictor
    predictor = initialize_sam()
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Predict masks
    masks, mask_scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # Get the best mask based on score
    best_mask_index = np.argmax(mask_scores)
    mask = masks[best_mask_index].astype(np.uint8)
    
    # Create a colored visualization of the mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [0, 255, 0]  # Green color for the mask
    
    # Save mask if output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
    return colored_mask

if __name__ == "__main__":
    # Example usage
    ref_image_path = "/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/loftbed.png"

    image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    h, w = image.shape[0], image.shape[1]
    # point_coords = [[500, 500]]  # Example point coordinates [x, y]
    # point_labels = [1]  # 1 for foreground point
    ref_object_location = [[0.227, 0.111], [0.182, 0.407], [0.309, 0.495], [0.396, 0.414], [0.394, 0.497], [0.223, 0.568], [0.444, 0.694], [0.545, 0.616], [0.522, 0.846], [0.535, 0.922], [0.628, 0.693], [0.806, 0.442], [0.829, 0.351], [0.55, 0.23], [0.562, 0.12], [0.607, 0.172], [0.285, 0.178], [0.828, 0.587], [0.861, 0.116], [0.777, 0.286]]
    point_coords = np.array([[h*loc[1], w*loc[0]] for loc in ref_object_location])
    point_labels = np.ones(len(ref_object_location))
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path
    output_path = os.path.join(output_dir, "predicted_mask.png")
    
    # Predict mask
    mask = predict_mask(ref_image_path, point_coords, point_labels, output_path)
    
    print(f"Mask saved to: {output_path}")
