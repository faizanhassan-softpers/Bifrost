import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# Load the model
sam = sam_model_registry["vit_h"](checkpoint="/home/ec2-user/SageMaker/model_weights/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Load and prepare the image
ref_image_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/object.jpg'
image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
h, w = image.shape[0], image.shape[1]

# Define point prompt
ref_object_location = [0.5, 0.45]
point_coords = np.array([[h * ref_object_location[1], w * ref_object_location[0]]])
point_labels = np.array([1])

# Set the image and predict masks
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)

# Save all masks
for i, mask in enumerate(masks):
    mask_uint8 = (mask * 255).astype(np.uint8)  # convert binary mask to 0-255 for saving
    cv2.imwrite(f"./mask.png", mask_uint8)
