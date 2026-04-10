import cv2
import numpy as np
import os
import glob

def process_all_census_images():
    """Process all census images through the complete pipeline"""
    
    print("=== BATCH PROCESSING ALL CENSUS IMAGES ===")
    
    # Get all raw images from Jeremy
    image_paths = glob.glob('data/from_jeremy/images/*.jpg')
    print(f"Found {len(image_paths)} census images to process")
    
    # Process each image
    successful = 0
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)} ---")
        print(f"File: {os.path.basename(image_path)}")
        
        try:
            # Step 1: Preprocessing
            print("  Preprocessing...")
            preprocessed = preprocess_single_image(image_path)
            
            # Step 2: Smart adaptive extraction
            print("  Extracting cells...")
            cells_extracted = extract_cells_from_image(preprocessed, i)
            
            if cells_extracted > 0:
                successful += 1
                print(f"   Success: {cells_extracted} cells")
            else:
                print(f"  Failed to extract cells")
                
        except Exception as e:
            print(f" Error: {e}")
    
    print(f"\n BATCH PROCESSING COMPLETE!")
    print(f"Successfully processed {successful}/{len(image_paths)} images")
    print(f"Extracted data saved to: data/extracted_data/")

def preprocess_single_image(image_path):
    """Preprocess a single census image"""
    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply the same preprocessing as preprocess_bw.py
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    return binary

def extract_cells_from_image(image, image_index):
    """Extract cells from a single image using smart adaptive method"""
    
    # Use the same column coordinates that worked for your first image
    columns = {
        'name': (1220, 1885),
        'race': (2285, 2388),
        'age': (2388, 2488),
        'occupation': (5240, 5620),
        'street': (633, 712)
    }
    
    first_row_y = 1263
    expected_row_height = 78
    num_rows = 40
    
    # Create output directory for this image
    base_name = f"image_{image_index:03d}"
    output_dir = f"data/extracted_data/{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple row boundaries (we can add the smart adaptation later)
    row_boundaries = [first_row_y + i * expected_row_height for i in range(num_rows + 1)]
    
    # Extract cells
    cell_count = 0
    for row_idx in range(num_rows):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        
        for col_name, (x1, x2) in columns.items():
            cell_img = image[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                filename = f"{base_name}_row{row_idx:02d}_{col_name}.png"
                cv2.imwrite(f"{output_dir}/{filename}", cell_img)
                cell_count += 1
    
    return cell_count

if __name__ == "__main__":
    process_all_census_images()