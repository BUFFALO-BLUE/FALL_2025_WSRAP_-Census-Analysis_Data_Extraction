# scripts/batch_extraction_deskewed.py
import cv2
import numpy as np
import os
from pathlib import Path
import time
import json

def extract_from_deskewed_image(image_path, output_dir):
    """
    Extract cells from a single deskewed census image.
    Returns: number of cells extracted
    """
    # Read the deskewed image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load {image_path}")
        return 0
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # EXACT COLUMN COORDINATES (from your working extraction)
    # These should work better with deskewed images
    columns = {
        'street': (629, 718),
        'house_number': (718, 836),
        'rented_owned': (914, 994),
        'price_rent': (996, 1143),
        'head': (1889, 2204),       # Column 5 (Head of Household)
        'gender': (2204, 2285),     # Column 6
        'race': (2285, 2388),       # Column 7
        'marital_status': (2491, 2574),  # Column 8
        'hours_worked': (4939, 5092),    # Column 24
        'wages': (6433, 6588)       # Column 26
    }
    
    # ROW DETECTION - Adaptive for deskewed images
    # Project pixel intensities horizontally
    horizontal_projection = np.sum(gray, axis=1)
    
    # Find valleys (spaces between rows)
    row_boundaries = []
    threshold = np.mean(horizontal_projection) * 0.7  # 70% of mean
    
    for i in range(1, len(horizontal_projection)-1):
        if (horizontal_projection[i] < threshold and 
            horizontal_projection[i-1] >= threshold):
            row_boundaries.append(i)
    
    # If not enough rows found, use fixed spacing
    if len(row_boundaries) < 20:
        print(f"   ⚠️  Only found {len(row_boundaries)} rows, using fixed spacing")
        first_row_y = 1263
        expected_row_height = 78
        row_boundaries = [first_row_y + i * expected_row_height 
                         for i in range(41)]  # 40 rows + end
    
    # Extract cells for first 40 rows
    cells_extracted = 0
    image_name = Path(image_path).stem
    
    for row_idx in range(min(40, len(row_boundaries)-1)):
        y_start = row_boundaries[row_idx]
        y_end = row_boundaries[row_idx + 1]
        
        # Skip if row is too small
        if y_end - y_start < 20:
            continue
        
        # Check if this is a head row (using head column content)
        head_cell = gray[y_start:y_end, 1889:2204]
        if head_cell.size > 0:
            black_pixels = np.count_nonzero(head_cell < 128)  # Threshold for "ink"
            total_pixels = head_cell.shape[0] * head_cell.shape[1]
            ink_percentage = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            is_head = 5 < ink_percentage < 60  # Head rows have some writing
            row_type = "head" if is_head else "non_head"
        else:
            row_type = "non_head"
            is_head = False
        
        # Extract all columns for this row
        for col_name, (x_start, x_end) in columns.items():
            # Skip non-head rows for head-specific columns if desired
            # if not is_head and col_name in ['head', 'race', 'gender']:
            #     continue
            
            cell_img = image[y_start:y_end, x_start:x_end]
            
            if cell_img.size > 0:
                # Create filename
                prefix = "HEAD_" if is_head else "NONHEAD_"
                filename = f"{prefix}row{row_idx:02d}_{col_name}.png"
                
                # Save path
                save_dir = output_dir / row_type + "_rows"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename
                
                # Save the cell
                cv2.imwrite(str(save_path), cell_img)
                cells_extracted += 1
    
    return cells_extracted

def batch_extract_from_deskewed():
    """Process ALL deskewed images"""
    
    input_dir = Path("data/from_jeremy/images_deskewed")
    output_root = Path("data/extracted_cells_deskewed")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Check if deskewed images exist
    if not input_dir.exists():
        print(f"❌ ERROR: Deskewed images not found at {input_dir}")
        print(f"   Run deskewing script first: python scripts/enhanced_deskew_only.py")
        return None
    
    # Get all deskewed images
    all_images = sorted(list(input_dir.glob("*.jpg")))
    
    if not all_images:
        print(f"❌ No images found in {input_dir}")
        return None
    
    print(f"📊 EXTRACTING FROM {len(all_images)} DESKEWED IMAGES")
    print("="*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_root}")
    print("="*60)
    
    # Progress tracking
    progress_file = output_root / "extraction_progress.json"
    
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        processed = set(progress.get('processed', []))
    else:
        progress = {'processed': [], 'cells_per_image': {}}
        processed = set()
    
    # Statistics
    total_cells = 0
    start_time = time.time()
    
    # Process each image
    for i, img_path in enumerate(all_images):
        if img_path.name in processed:
            continue
            
        print(f"\n[{i+1}/{len(all_images)}] {img_path.name}")
        
        # Create output folder for this image
        image_output_dir = output_root / img_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract cells
        cells_count = extract_from_deskewed_image(img_path, image_output_dir)
        total_cells += cells_count
        
        # Update progress
        processed.add(img_path.name)
        progress['processed'] = list(processed)
        progress['cells_per_image'][img_path.name] = cells_count
        
        # Save progress every 10 images
        if (i + 1) % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            # Calculate speed
            elapsed_minutes = (time.time() - start_time) / 60
            speed = (i + 1) / elapsed_minutes if elapsed_minutes > 0 else 0
            png_speed = total_cells / elapsed_minutes if elapsed_minutes > 0 else 0
            
            print(f"\n📈 Progress: {i+1}/{len(all_images)}")
            print(f"   Images/minute: {speed:.1f}")
            print(f"   PNGs/minute: {png_speed:.1f}")
            print(f"   Total cells: {total_cells:,}")
    
    # Save final progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Create summary
    create_extraction_summary(progress, output_root, start_time, len(all_images))
    
    return output_root

def create_extraction_summary(progress, output_dir, start_time, total_images):
    """Create summary of extraction results"""
    
    total_time = (time.time() - start_time) / 60  # minutes
    processed_count = len(progress['processed'])
    cells_per_image = progress['cells_per_image']
    
    if cells_per_image:
        avg_cells = sum(cells_per_image.values()) / len(cells_per_image)
        max_cells = max(cells_per_image.values())
        min_cells = min(cells_per_image.values())
        total_cells = sum(cells_per_image.values())
    else:
        avg_cells = max_cells = min_cells = total_cells = 0
    
    summary = f"""EXTRACTION SUMMARY
===================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total images available: {total_images}
Images processed: {processed_count} ({processed_count/total_images*100:.1f}%)
Total cells extracted: {total_cells:,}
Average cells per image: {avg_cells:.0f}
Maximum cells from one image: {max_cells}
Minimum cells from one image: {min_cells}
Total time: {total_time:.1f} minutes
Processing speed: {processed_count/total_time:.1f} images/minute
PNG creation speed: {total_cells/total_time:.1f} PNGs/minute

FILE STRUCTURE:
---------------
Each image folder contains:
  {output_dir}/[image_name]/
    ├── head_rows/       # Head household rows
    └── non_head_rows/   # Other household members

Each PNG file is named:
  HEAD_row00_race.png
  HEAD_row00_house_number.png
  NONHEAD_row10_race.png
  etc.

NEXT STEPS:
-----------
1. Verify extraction quality
2. Map extracted cells to Excel data
3. Create training dataset
4. Train OCR model

SAMPLE OUTPUT PATHS:
--------------------
"""
    
    # Add sample paths
    sample_dirs = list(output_dir.glob("*/head_rows"))
    if sample_dirs:
        sample_files = list(sample_dirs[0].glob("*.png"))[:5]
        for file in sample_files:
            summary += f"  {file}\n"
    
    # Save summary
    summary_path = output_dir / "extraction_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Summary saved: {summary_path}")
    print(f"Images processed: {processed_count}/{total_images}")
    print(f"Total cells: {total_cells:,}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Speed: {total_cells/total_time:.1f} PNGs/minute")

if __name__ == "__main__":
    print("="*60)
    print("📝 CENSUS CELL EXTRACTION FROM DESKEWED IMAGES")
    print("="*60)
    
    print("\nPREREQUISITE:")
    print("  First run: python scripts/enhanced_deskew_only.py")
    print("  (This creates 'data/from_jeremy/images_deskewed/')")
    
    print("\nThis script will:")
    print("1. Extract cells from all deskewed images")
    print("2. Save 400+ PNGs per image")
    print("3. Organize by image and row type")
    
    print(f"\n⚠️  Will create ~2 MILLION PNG files")
    print("   (~400 cells × 5,000 images)")
    print("   Ensure you have ~10GB free space")
    
    # Test if deskewed images exist
    deskewed_dir = Path("data/from_jeremy/images_deskewed")
    if not deskewed_dir.exists():
        print(f"\n❌ ERROR: {deskewed_dir} not found!")
        print("   Run deskewing script first.")
        exit(1)
    
    deskewed_count = len(list(deskewed_dir.glob("*.jpg")))
    print(f"\n✅ Found {deskewed_count} deskewed images")
    
    confirm = input("\nStart extraction? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        output_dir = batch_extract_from_deskewed()
        
        if output_dir:
            print(f"\n🎯 NEXT STEP:")
            print(f"Count extracted PNGs:")
            print(f"   find {output_dir} -name '*.png' | wc -l")
            
            print(f"\n🔍 Sample check:")
            print(f"   ls {output_dir}/m-t0627-00538-00634/head_rows/ | head -10")
    else:
        print("Extraction cancelled.")