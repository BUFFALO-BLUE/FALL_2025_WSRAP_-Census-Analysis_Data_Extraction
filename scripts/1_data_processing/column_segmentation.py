import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_lines(image_path):
    """Detect horizontal and vertical lines in the preprocessed image"""
    
    # Load the preprocessed image
    image = cv2.imread(image_path, 0)  # Read as grayscale
    if image is None:
        print("Error: Could not load image. Check the path.")
        return None
    
    binary = image.copy()
    
    # Create images for horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines  
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine both line images
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    
    return image, horizontal_lines, vertical_lines, table_structure

def visualize_results(original, horizontal, vertical, structure):
    """Display the results"""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Processed Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(horizontal, cmap='gray')
    plt.title('Detected Horizontal Lines')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(vertical, cmap='gray')
    plt.title('Detected Vertical Lines')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(structure, cmap='gray')
    plt.title('Combined Table Structure')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Use the preprocessed image from the previous step
    input_path = os.path.join('data', 'processed', 'preprocessed_image.png')
    
    print("Starting column segmentation...")
    results = detect_lines(input_path)
    
    if results is not None:
        original, horizontal, vertical, structure = results
        visualize_results(original, horizontal, vertical, structure)
        print("Line detection completed successfully!")
        
        # Save the results for future use
        cv2.imwrite(os.path.join('data', 'processed', 'horizontal_lines.png'), horizontal)
        cv2.imwrite(os.path.join('data', 'processed', 'vertical_lines.png'), vertical)
        cv2.imwrite(os.path.join('data', 'processed', 'table_structure.png'), structure)
        print("Results saved to data/processed/")