import cv2
import numpy as np
import os
from pathlib import Path

def extract_and_classify_mask(type,image_path, output_dir):
   

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, type), exist_ok=True)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Red color in HSV (note: red wraps around in HSV, so we need two ranges)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    

    # Process the image
    # Check if there are any red pixels (kanama)
    if np.any(mask_red):
        # Create binary mask for YOLOv8
        mask = np.zeros_like(mask_red)
        mask[mask_red > 0] = 255
        
        # Save the mask
        base_filename = Path(image_path).stem
        
        output_path = os.path.join(output_dir, type, f"{base_filename}.png")
        cv2.imwrite(output_path, mask)


    

# Example usage
if __name__ == "__main__":
    # Paths
    image_path = "Iskemi Veri Seti\Overlay"  
    mask_path = "Iskemi Veri Seti\MASK"  
    
    #tüm resimleri döngüye al
    image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
    for image_file in image_files:
        full_image_path = os.path.join(image_path, image_file)
        # print(f"Processing {full_image_path}")
        
        # Extract and classify masks
        extract_and_classify_mask("iskemi",full_image_path, mask_path)

