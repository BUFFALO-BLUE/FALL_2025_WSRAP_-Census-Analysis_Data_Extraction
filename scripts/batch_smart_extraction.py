import cv2
import numpy as np
import os
import glob
from pathlib import Path

def batch_smart_extraction(use_aligned=True, use_ready=False):
    """Process ALL census images with smart adaptive extraction
    
    Parameters:
    -----------
    use_aligned : bool
        If True, use images from images_aligned_to_first folder
        If False, use images from original images folder
    use_ready : bool
        If True, use images from images_ready_for_extraction folder
        If False, use either aligned or original images
    """
    
    print("=== BATCH SMART EXTRACTION ===")
    print(f"Mode: {'ALIGNED' if use_aligned else 'ORIGINAL'}{' (READY)' if use_ready else ''}")
    
    # Determine which folder to use
    if use_ready:
        # Use pre-processed ready images
        image_dir = 'data/from_jeremy/images_ready_for_extraction'
        output_suffix = 'ready'
    elif use_aligned:
        # Use deskewed/aligned images
        image_dir = 'data/from_jeremy/images_aligned_to_first'
        output_suffix = 'aligned'
    else:
        # Use original images
        image_dir = 'data/from_jeremy/images'
        output_suffix = 'original'
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        print("Please run alignment first or check the folder structure.")
        return 0, 0
    
    # Get all images
    image_paths = glob.glob(f'{image_dir}/*.jpg')
    
    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        return 0, 0
    
    print(f"Found {len(image_paths)} census images in {image_dir}")
    
    total_head_rows = 0
    total_cells = 0
    processed_images = []
    failed_images = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)} ---")
        print(f"File: {os.path.basename(image_path)}")
        
        try:
            # Load and preprocess
            image = cv2.imread(image_path, 0)  # Read as grayscale
            if image is None:
                print(f"❌ Could not load {image_path}")
                failed_images.append(os.path.basename(image_path))
                continue
            
            # Preprocess
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Extract using smart method
            head_rows, cells_extracted = extract_smart_from_image(
                binary, i, os.path.basename(image_path), output_suffix
            )
            
            total_head_rows += head_rows
            total_cells += cells_extracted
            processed_images.append(os.path.basename(image_path))
            
            print(f"✅ {head_rows} head rows, {cells_extracted} cells extracted")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_images.append(os.path.basename(image_path))
    
    print(f"\n{'='*60}")
    print("🎉 BATCH PROCESSING COMPLETE!")
    print(f"{'='*60}")
    
    # Summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"Input folder: {image_dir}")
    print(f"Images processed successfully: {len(processed_images)}")
    print(f"Images failed: {len(failed_images)}")
    print(f"Total head rows across all images: {total_head_rows}")
    print(f"Total cells extracted: {total_cells}")
    
    if failed_images:
        print(f"\n❌ Failed images:")
        for img in failed_images:
            print(f"  - {img}")
    
    # Save processing log
    log_dir = Path("data/processing_logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"extraction_log_{output_suffix}.txt"
    with open(log_file, 'w') as f:
        f.write(f"EXTRACTION PROCESSING LOG - {output_suffix.upper()} IMAGES\n")
        f.write("="*50 + "\n")
        f.write(f"Input folder: {image_dir}\n")
        f.write(f"Total images found: {len(image_paths)}\n")
        f.write(f"Successfully processed: {len(processed_images)}\n")
        f.write(f"Failed: {len(failed_images)}\n")
        f.write(f"Total head rows: {total_head_rows}\n")
        f.write(f"Total cells extracted: {total_cells}\n\n")
        
        f.write("SUCCESSFULLY PROCESSED IMAGES:\n")
        f.write("-"*30 + "\n")
        for img in processed_images:
            f.write(f"{img}\n")
        
        if failed_images:
            f.write("\nFAILED IMAGES:\n")
            f.write("-"*30 + "\n")
            for img in failed_images:
                f.write(f"{img}\n")
    
    print(f"\n📁 Output organized by image in: data/extracted_cells_{output_suffix}/")
    print(f"📝 Processing log saved to: {log_file}")
    print(f"🔧 Next step: Run Excel mapping script")
    
    return total_head_rows, total_cells

def extract_smart_from_image(image, image_index, image_name, output_suffix):
    """Extract from a single image using smart adaptive method"""
    
    # ALL REQUIRED COLUMNS
    columns = {
        'house_number': (718, 836),
        'rented_owned': (914, 994),
        'price_rent': (996, 1143),
        'head': (1889, 2204),
        'gender': (2204, 2285),
        'race': (2285, 2388),
        'marital_status': (2491, 2574),
        'hours_worked': (4939, 5092),
        'wages': (6433, 6588)
    }
    
    # Parameters - adjust if needed for deskewed images
    first_row_y = 1263
    expected_row_height = 78
    num_rows = 40
    
    # Create unique output folder with suffix
    base_name = os.path.splitext(image_name)[0]
    output_dir = f"data/extracted_cells_{output_suffix}/{base_name}"
    head_output_dir = f"{output_dir}/head_rows"
    non_head_output_dir = f"{output_dir}/non_head_rows"
    
    # Create directories
    os.makedirs(head_output_dir, exist_ok=True)
    os.makedirs(non_head_output_dir, exist_ok=True)
    
    # Simple row boundaries
    row_boundaries = [first_row_y + i * expected_row_height for i in range(num_rows + 1)]
    
    # Optional: Save a preview of the extraction area for debugging
    debug_dir = Path(f"data/debug_extraction_{output_suffix}")
    debug_dir.mkdir(exist_ok=True)
    
    # Create a color preview of the extraction area
    if image_index < 3:  # Only for first 3 images
        color_preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw extraction area
        for row_idx in range(num_rows):
            y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
            
            # Draw row boundaries
            cv2.line(color_preview, (0, y1), (image.shape[1], y1), (0, 255, 0), 1)
            
            # Draw column boundaries
            for col_name, (x1, x2) in columns.items():
                cv2.rectangle(color_preview, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Label columns in first row
                if row_idx == 0:
                    cv2.putText(color_preview, col_name[:5], (x1+5, y1+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.imwrite(str(debug_dir / f"{base_name}_extraction_preview.jpg"), color_preview)
    
    # Extract cells
    head_rows_count = 0
    cells_extracted = 0
    
    for row_idx in range(num_rows):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        
        # Check head column
        head_cell = image[y1:y2, 1889:2204]
        
        # Skip if empty
        if head_cell.size == 0:
            continue
            
        black_pixels = np.count_nonzero(head_cell == 0)
        total_pixels = head_cell.shape[0] * head_cell.shape[1]
        black_percentage = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Adjusted threshold for deskewed images
        is_head = 5 < black_percentage < 60  # Slightly more lenient for deskewed
        
        if is_head:
            head_rows_count += 1
        
        # Extract all columns
        for col_name, (x1, x2) in columns.items():
            cell_img = image[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                prefix = "HEAD_" if is_head else ""
                filename = f"{prefix}row{row_idx:02d}_{col_name}.png"
                
                # Save to appropriate directory
                save_path = os.path.join(head_output_dir if is_head else non_head_output_dir, filename)
                cv2.imwrite(save_path, cell_img)
                
                if is_head:
                    cells_extracted += 1
    
    return head_rows_count, cells_extracted

def compare_extraction_modes():
    """Compare extraction results from different image sources"""
    print("="*60)
    print(" COMPARING EXTRACTION MODES")
    print("="*60)
    
    modes = [
        ("Original images", False, False),
        ("Aligned images", True, False),
    ]
    
    results = {}
    
    for mode_name, use_aligned, use_ready in modes:
        print(f"\n Processing {mode_name}...")
        head_rows, total_cells = batch_smart_extraction(
            use_aligned=use_aligned,
            use_ready=use_ready
        )
        results[mode_name] = {
            'head_rows': head_rows,
            'total_cells': total_cells
        }
    
    # Print comparison
    print(f"\n{'='*60}")
    print(" COMPARISON RESULTS")
    print("="*60)
    
    print(f"\n{'Mode':<20} {'Head Rows':<12} {'Total Cells':<12}")
    print("-"*44)
    
    for mode_name, data in results.items():
        print(f"{mode_name:<20} {data['head_rows']:<12} {data['total_cells']:<12}")
    
    return results

if __name__ == "__main__":
    print("="*70)
    print(" BATCH SMART EXTRACTION - MULTIPLE MODES")
    print("="*70)
    
    print("\nChoose extraction mode:")
    print("1. Use deskewed/aligned images (recommended after alignment)")
    print("2. Use original images")
    print("3. Compare both modes")
    print("4. Use ready-for-extraction images")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Use aligned images
        batch_smart_extraction(use_aligned=True, use_ready=False)
    elif choice == '2':
        # Use original images
        batch_smart_extraction(use_aligned=False, use_ready=False)
    elif choice == '3':
        # Compare both
        compare_extraction_modes()
    elif choice == '4':
        # Use ready images
        batch_smart_extraction(use_aligned=False, use_ready=True)
    else:
        print("Invalid choice. Defaulting to aligned images.")
        batch_smart_extraction(use_aligned=True, use_ready=False)