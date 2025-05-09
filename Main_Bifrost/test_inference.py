import os
import cv2
from run_inference_lib import run_inference

def main():
    # Create temp directory if it doesn't exist
    temp_dir = './examples/temp'
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'DEPTH'), exist_ok=True)

    # Test parameters
    bg_image_path = '/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/background.jpg'
    bg_mask = [0.153, 0.18, 0.847, 0.64]  # [x, y, w, h] in range [0, 1]
    ref_object_location = [[0.227, 0.111], [0.182, 0.407], [0.309, 0.495], [0.396, 0.414], [0.394, 0.497], [0.223, 0.568], [0.444, 0.694], [0.545, 0.616], [0.522, 0.846], [0.535, 0.922], [0.628, 0.693], [0.806, 0.442], [0.829, 0.351], [0.55, 0.23], [0.562, 0.12], [0.607, 0.172], [0.285, 0.178], [0.828, 0.587], [0.861, 0.116], [0.777, 0.286]]

    # Run inference
    print("Running inference...")
    gen_image, vis_image = run_inference(
        temp_dir_path=temp_dir,
        bg_image_path=bg_image_path,

        ref_object_location=ref_object_location,
        # Optional parameters with default values
        bg_object_location=[0.6, 0.5],
        depth=[0.1, 0.22],
        pixel_num=0.02,
        mode='place',
        flip_image=False,
        sobel_color=False,
        sobel_threshold=50
    )

    if gen_image is not None and vis_image is not None:
        # Save results
        cv2.imwrite(os.path.join(temp_dir, 'test_result.jpg'), gen_image[:,:,::-1])
        cv2.imwrite(os.path.join(temp_dir, 'test_visualization.jpg'), vis_image[:,:,::-1])
        print("Results saved in:", temp_dir)
    else:
        print("Inference failed - check if mask coordinates are valid")

if __name__ == '__main__':
    main() 