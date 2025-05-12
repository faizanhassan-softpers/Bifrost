import cv2
import numpy as np

def draw_points_on_image(image_path, points, output_path=None, point_size=5, point_color=(0, 255, 0)):
    """
    Draw points on an image and optionally save the result.
    
    Args:
        image_path (str): Path to the input image
        points (list): List of normalized (x, y) coordinates to draw (values between 0 and 1)
        output_path (str, optional): Path to save the output image. If None, displays the image
        point_size (int): Size of the points to draw
        point_color (tuple): BGR color tuple for the points (default: green)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    print(f"Image loaded successfully. Shape: {image.shape}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Draw each point
    for i, (x, y) in enumerate(points):
        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(x * width)
        pixel_y = int(y * height)
        print(f"Point {i}: Normalized ({x:.3f}, {y:.3f}) -> Pixel ({pixel_x}, {pixel_y})")
        cv2.circle(image, (pixel_x, pixel_y), point_size, point_color, -1)
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved result to {output_path}")
    else:
        cv2.imshow('Image with Points', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Image path
    image_path = "/Users/admin/dev/Bifrost-parent/Bifrost/Main_Bifrost/examples/TEST/Input/loftbed.png"
    
    # Points in normalized coordinates (0-1)
    points = [
        [0.227, 0.111], [0.182, 0.407], [0.309, 0.495], [0.396, 0.414], [0.394, 0.497],
        [0.223, 0.568], [0.444, 0.694], [0.545, 0.616], [0.522, 0.846], [0.535, 0.922],
        [0.628, 0.693], [0.806, 0.442], [0.829, 0.351], [0.55, 0.23], [0.562, 0.12],
        [0.607, 0.172], [0.285, 0.178], [0.828, 0.587], [0.861, 0.116], [0.777, 0.286]
    ]
    
    # Draw points on the image with larger points and red color for better visibility
    draw_points_on_image(
        image_path,
        points,
        point_size=30,  # Increased point size
        point_color=(0, 0, 255)  # Red color in BGR for better visibility
    )

if __name__ == '__main__':
    main() 