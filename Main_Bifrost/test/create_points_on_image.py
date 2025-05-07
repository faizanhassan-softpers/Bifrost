import cv2
import numpy as np

def draw_point_on_image(image, point, color=(0, 255, 0), radius=5):
    """
    Draw a point on an image at the specified coordinates.
    
    Args:
        image: Input image (numpy array)
        point: Tuple or list containing normalized coordinates [x, y] between 0 and 1
        color: BGR color tuple (default: green)
        radius: Radius of the point in pixels (default: 5)
    
    Returns:
        Image with the point drawn on it
    """
    # Make a copy of the image to avoid modifying the original
    image_copy = image.copy()
    
    # Get image dimensions
    height, width = image_copy.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x = int(point[0] * width)
    y = int(point[1] * height)
    
    # Draw the point as a filled circle
    cv2.circle(image_copy, (x, y), radius, color, -1)
    
    return image_copy

# Example usage
if __name__ == "__main__":
    # Read image from file path
    image_path = "/Users/admin/dev/Bifrost-parent/Bifrost/Main_Bifrost/examples/TEST/Input/background.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        exit()
    
    # Draw a point at normalized coordinates (0.6, 0.5)
    point = [0.6, 0.5]
    result_image = draw_point_on_image(image, point)
    
    # Display the image
    cv2.imshow("Image with Point", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the result
    output_path = "result_with_point.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Saved result to {output_path}")
