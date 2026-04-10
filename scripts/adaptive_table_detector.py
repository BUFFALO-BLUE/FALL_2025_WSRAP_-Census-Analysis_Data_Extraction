# scripts/adaptive_table_detector.py
"""
Finds the table grid structure ADAPTIVELY in each census image.
Detects rows and columns automatically - no hard-coded coordinates!
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import json

class AdaptiveTableDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.debug_dir = Path("data/debug/table_detection")
        if debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_table_grid(self, image_path):
        """
        Detect table grid structure (rows and columns) in an image.
        Returns: dict with row_boundaries, column_boundaries, and success flag
        """
        print(f" Detecting table grid in: {Path(image_path).name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f" Could not load image")
            return None
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for better binarization
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to enhance table structure
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        # Find horizontal lines (rows)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
        
        # Find vertical lines (columns)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
        
        if self.debug:
            cv2.imwrite(str(self.debug_dir / "1_binary.png"), binary)
            cv2.imwrite(str(self.debug_dir / "2_horizontal_lines.png"), horizontal)
            cv2.imwrite(str(self.debug_dir / "3_vertical_lines.png"), vertical)
        
        # Detect rows
        row_boundaries = self._detect_rows(horizontal, gray)
        
        # Detect columns
        column_boundaries = self._detect_columns(vertical, gray)
        
        if self.debug:
            # Visualize detected grid
            grid_viz = img.copy()
            
            # Draw row boundaries
            for y in row_boundaries:
                cv2.line(grid_viz, (0, y), (img.shape[1], y), (0, 255, 0), 2)
            
            # Draw column boundaries
            for x in column_boundaries:
                cv2.line(grid_viz, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
            
            # Label rows and columns
            for i, y in enumerate(row_boundaries[:-1]):
                cv2.putText(grid_viz, f"Row {i}", (50, y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for i, x in enumerate(column_boundaries[:-1]):
                cv2.putText(grid_viz, f"Col {i}", (x + 10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imwrite(str(self.debug_dir / "4_detected_grid.png"), grid_viz)
        
        # Check if detection was successful
        success = len(row_boundaries) >= 5 and len(column_boundaries) >= 5
        
        result = {
            'image_path': str(image_path),
            'image_shape': img.shape,
            'row_boundaries': row_boundaries,
            'column_boundaries': column_boundaries,
            'num_rows': len(row_boundaries) - 1,
            'num_columns': len(column_boundaries) - 1,
            'success': success,
            'average_row_height': self._calculate_average_height(row_boundaries) if len(row_boundaries) > 1 else 0,
            'average_column_width': self._calculate_average_width(column_boundaries) if len(column_boundaries) > 1 else 0
        }
        
        return result
    
    def _detect_rows(self, horizontal_lines, original_gray):
        """
        Detect row boundaries from horizontal lines.
        """
        # Project horizontal lines to get row boundaries
        row_projection = np.sum(horizontal_lines, axis=1)
        
        # Find peaks in projection (these are likely row separators)
        peaks = self._find_peaks(row_projection, min_distance=30, threshold=1000)
        
        # Sort and ensure we have enough rows (typically 40+ rows in census)
        peaks = sorted(peaks)
        
        if len(peaks) < 10:  # Not enough rows detected
            # Fallback: estimate rows based on content
            peaks = self._estimate_rows_from_content(original_gray)
        
        # Add top and bottom boundaries
        if peaks:
            # Ensure we have boundaries at top and bottom
            boundaries = [0] + peaks + [original_gray.shape[0]]
            
            # Remove duplicates and sort
            boundaries = sorted(list(set(boundaries)))
            
            # Filter boundaries that are too close together
            filtered_boundaries = [boundaries[0]]
            for i in range(1, len(boundaries)):
                if boundaries[i] - filtered_boundaries[-1] > 20:  # Min row height 20px
                    filtered_boundaries.append(boundaries[i])
            
            return filtered_boundaries
        
        return [0, original_gray.shape[0]]  # Fallback
    
    def _detect_columns(self, vertical_lines, original_gray):
        """
        Detect column boundaries from vertical lines.
        """
        # Project vertical lines to get column boundaries
        col_projection = np.sum(vertical_lines, axis=0)
        
        # Find peaks in projection (these are likely column separators)
        peaks = self._find_peaks(col_projection, min_distance=50, threshold=500)
        
        # Sort peaks
        peaks = sorted(peaks)
        
        if len(peaks) < 5:  # Not enough columns detected
            # Fallback: estimate columns based on common census layout
            peaks = self._estimate_columns_from_layout(original_gray)
        
        # Add left and right boundaries
        if peaks:
            boundaries = [0] + peaks + [original_gray.shape[1]]
            
            # Remove duplicates and sort
            boundaries = sorted(list(set(boundaries)))
            
            # Filter boundaries that are too close together
            filtered_boundaries = [boundaries[0]]
            for i in range(1, len(boundaries)):
                if boundaries[i] - filtered_boundaries[-1] > 30:  # Min column width 30px
                    filtered_boundaries.append(boundaries[i])
            
            return filtered_boundaries
        
        return [0, original_gray.shape[1]]  # Fallback
    
    def _find_peaks(self, data, min_distance=20, threshold=None):
        """
        Find peaks in 1D data.
        """
        if threshold is None:
            threshold = np.mean(data) * 0.5
        
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if data[i] > threshold:
                # Check if it's a local maximum
                is_peak = True
                for j in range(1, min_distance + 1):
                    if data[i] < data[i - j] or data[i] < data[i + j]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
        
        return peaks
    
    def _estimate_rows_from_content(self, gray_image):
        """
        Estimate row boundaries based on text content (fallback).
        """
        height, width = gray_image.shape
        
        # Horizontal projection of text
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Sum along rows to find text density
        row_density = np.sum(binary, axis=1) / 255
        
        # Find valleys between text rows
        valleys = []
        for i in range(1, len(row_density) - 1):
            if row_density[i] < np.mean(row_density) * 0.3:  # Valley
                valleys.append(i)
        
        # Group valleys and take average
        if valleys:
            # Simple grouping
            grouped_valleys = []
            current_group = [valleys[0]]
            
            for i in range(1, len(valleys)):
                if valleys[i] - valleys[i-1] < 50:  # Group valleys close together
                    current_group.append(valleys[i])
                else:
                    grouped_valleys.append(int(np.mean(current_group)))
                    current_group = [valleys[i]]
            
            if current_group:
                grouped_valleys.append(int(np.mean(current_group)))
            
            return grouped_valleys
        
        # Fallback: estimate 40 equal rows
        return list(range(0, height, height // 40))[1:-1]
    
    def _estimate_columns_from_layout(self, gray_image):
        """
        Estimate column boundaries based on common census layout (fallback).
        """
        width = gray_image.shape[1]
        
        # Common census column ratios (approximate)
        column_ratios = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        peaks = [int(width * ratio) for ratio in column_ratios]
        
        return peaks
    
    def _calculate_average_height(self, boundaries):
        """Calculate average row height."""
        heights = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        return np.mean(heights) if heights else 0
    
    def _calculate_average_width(self, boundaries):
        """Calculate average column width."""
        widths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        return np.mean(widths) if widths else 0

def batch_adaptive_extraction():
    """
    Extract data from ALL images using adaptive table detection.
    No hard-coded coordinates - detects grid for each image!
    """
    
    print("="*70)
    print("ADAPTIVE TABLE EXTRACTION - SMART GRID DETECTION")
    print("="*70)
    
    # Choose input folder
    print("\nChoose input folder:")
    print("1. Deskewed/Aligned images (recommended)")
    print("2. Original images")
    print("3. Ready images")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        input_dir = Path("data/from_jeremy/images_aligned_to_first")
        output_suffix = "adaptive_aligned"
    elif choice == '2':
        input_dir = Path("data/from_jeremy/images")
        output_suffix = "adaptive_original"
    elif choice == '3':
        input_dir = Path("data/from_jeremy/images_ready_for_extraction")
        output_suffix = "adaptive_ready"
    else:
        print("Using aligned images by default")
        input_dir = Path("data/from_jeremy/images_aligned_to_first")
        output_suffix = "adaptive_aligned"
    
    if not input_dir.exists():
        print(f" Input directory not found: {input_dir}")
        return
    
    # Get all images
    image_paths = sorted(list(input_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f" No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Input: {input_dir}")
    
    # Create output directories
    output_dir = Path(f"data/extracted_cells_{output_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detector with debug for first few images
    detector = AdaptiveTableDetector(debug=True)
    
    # Process each image
    all_results = []
    extraction_stats = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_path.name}")
        
        try:
            # Detect table grid for this image
            grid_result = detector.detect_table_grid(img_path)
            
            if not grid_result or not grid_result['success']:
                print(f"  Table detection failed")
                all_results.append({
                    'image': img_path.name,
                    'success': False,
                    'error': 'Table detection failed'
                })
                continue
            
            # Extract data using detected grid
            stats = extract_with_detected_grid(img_path, grid_result, output_dir)
            
            all_results.append({
                'image': img_path.name,
                'success': True,
                'num_rows': grid_result['num_rows'],
                'num_columns': grid_result['num_columns'],
                'average_row_height': grid_result['average_row_height'],
                'average_column_width': grid_result['average_column_width']
            })
            
            extraction_stats.append(stats)
            
            print(f" Extracted: {stats['head_rows']} head rows, {stats['total_cells']} cells")
            
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({
                'image': img_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Save results and statistics
    save_extraction_results(all_results, extraction_stats, output_dir, output_suffix)
    
    return all_results

def extract_with_detected_grid(image_path, grid_result, output_dir):
    """
    Extract data using the detected grid structure.
    """
    img = cv2.imread(str(image_path), 0)  # Grayscale
    if img is None:
        return {'head_rows': 0, 'total_cells': 0}
    
    # Get grid boundaries
    row_boundaries = grid_result['row_boundaries']
    col_boundaries = grid_result['column_boundaries']
    
    # Identify key columns based on width (census columns have characteristic widths)
    # Head column is usually wide (contains names)
    # Other columns are narrower
    
    col_widths = [col_boundaries[i+1] - col_boundaries[i] for i in range(len(col_boundaries)-1)]
    avg_width = np.mean(col_widths)
    
    # Heuristic: Head column is usually 2-3x wider than average
    potential_head_cols = []
    for i, width in enumerate(col_widths):
        if width > avg_width * 1.8:
            potential_head_cols.append(i)
    
    # If we found potential head columns, use the widest one
    head_col_index = potential_head_cols[0] if potential_head_cols else -1
    
    # Create output folder for this image
    base_name = Path(image_path).stem
    image_output_dir = output_dir / base_name
    head_output_dir = image_output_dir / "head_rows"
    non_head_output_dir = image_output_dir / "non_head_rows"
    
    head_output_dir.mkdir(parents=True, exist_ok=True)
    non_head_output_dir.mkdir(parents=True, exist_ok=True)
    
    head_rows_count = 0
    total_cells = 0
    
    # Process each row
    for row_idx in range(len(row_boundaries) - 1):
        y1 = row_boundaries[row_idx]
        y2 = row_boundaries[row_idx + 1]
        row_height = y2 - y1
        
        # Skip very short rows (likely not data rows)
        if row_height < 20:
            continue
        
        # Check if this is a head row
        is_head = False
        if head_col_index >= 0:
            # Extract head cell
            x1_head = col_boundaries[head_col_index]
            x2_head = col_boundaries[head_col_index + 1]
            head_cell = img[y1:y2, x1_head:x2_head]
            
            # Check for content
            if head_cell.size > 0:
                black_pixels = np.sum(head_cell < 128)
                black_percentage = black_pixels / head_cell.size
                
                # Head cells typically have moderate text density
                if 0.05 < black_percentage < 0.7:
                    is_head = True
        
        if is_head:
            head_rows_count += 1
            output_subdir = head_output_dir
        else:
            output_subdir = non_head_output_dir
        
        # Extract all columns for this row
        for col_idx in range(len(col_boundaries) - 1):
            x1 = col_boundaries[col_idx]
            x2 = col_boundaries[col_idx + 1]
            
            cell_img = img[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                # Save cell
                prefix = "HEAD_" if is_head else ""
                filename = f"{prefix}row{row_idx:02d}_col{col_idx:02d}.png"
                save_path = output_subdir / filename
                cv2.imwrite(str(save_path), cell_img)
                
                if is_head:
                    total_cells += 1
    
    return {
        'head_rows': head_rows_count,
        'total_cells': total_cells,
        'num_rows_detected': len(row_boundaries) - 1,
        'num_cols_detected': len(col_boundaries) - 1
    }

def save_extraction_results(all_results, extraction_stats, output_dir, suffix):
    """
    Save extraction results and statistics.
    """
    # Create summary
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful
    
    total_head_rows = sum(s['head_rows'] for s in extraction_stats)
    total_cells = sum(s['total_cells'] for s in extraction_stats)
    
    # Save results JSON
    results_file = output_dir / f"extraction_results_{suffix}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(all_results),
                'successful': successful,
                'failed': failed,
                'total_head_rows': total_head_rows,
                'total_cells': total_cells
            },
            'results': all_results,
            'statistics': extraction_stats
        }, f, indent=2)
    
    # Save CSV summary
    import csv
    csv_file = output_dir / f"extraction_summary_{suffix}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Success', 'Rows', 'Columns', 'Head Rows', 'Cells Extracted'])
        
        for result, stats in zip(all_results, extraction_stats):
            writer.writerow([
                result['image'],
                'Yes' if result['success'] else 'No',
                result.get('num_rows', 0),
                result.get('num_columns', 0),
                stats.get('head_rows', 0),
                stats.get('total_cells', 0)
            ])
    
    print(f"\n{'='*70}")
    print(" ADAPTIVE EXTRACTION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n SUMMARY:")
    print(f"  Images processed: {len(all_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total head rows extracted: {total_head_rows}")
    print(f"  Total cells extracted: {total_cells}")
    
    print(f"\n Output saved to: {output_dir}")
    print(f" Results JSON: {results_file}")
    print(f" Summary CSV: {csv_file}")
    
    # Show successful detections
    print(f"\n Successful detections (first 5):")
    successful_results = [r for r in all_results if r['success']]
    for i, result in enumerate(successful_results[:5]):
        print(f"  {i+1}. {result['image']}: {result['num_rows']} rows, {result['num_columns']} columns")
    
    if failed > 0:
        print(f"\n Failed images:")
        for result in all_results:
            if not result['success']:
                print(f"  - {result['image']}: {result.get('error', 'Unknown error')}")

def test_single_image_detection(image_path):
    """
    Test table detection on a single image with visualization.
    """
    print(f"Testing table detection on: {Path(image_path).name}")
    
    detector = AdaptiveTableDetector(debug=True)
    result = detector.detect_table_grid(image_path)
    
    if result and result['success']:
        print(f"\n Table detected successfully!")
        print(f"   Image shape: {result['image_shape']}")
        print(f"   Rows detected: {result['num_rows']}")
        print(f"   Columns detected: {result['num_columns']}")
        print(f"   Average row height: {result['average_row_height']:.1f}px")
        print(f"   Average column width: {result['average_column_width']:.1f}px")
        
        # Create visualization
        img = cv2.imread(str(image_path))
        viz = img.copy()
        
        # Draw row boundaries
        for y in result['row_boundaries']:
            cv2.line(viz, (0, y), (img.shape[1], y), (0, 255, 0), 2)
        
        # Draw column boundaries
        for x in result['column_boundaries']:
            cv2.line(viz, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
        
        # Save visualization
        test_dir = Path("data/test_detection")
        test_dir.mkdir(exist_ok=True)
        
        viz_path = test_dir / f"{Path(image_path).stem}_grid.png"
        cv2.imwrite(str(viz_path), viz)
        
        print(f"\n Visualization saved to: {viz_path}")
        
        return result
    else:
        print(" Table detection failed")
        return None

if __name__ == "__main__":
    print("="*70)
    print(" SMART ADAPTIVE TABLE DETECTION & EXTRACTION")
    print("="*70)
    
    print("\nChoose mode:")
    print("1. Test table detection on a single image")
    print("2. Batch extract ALL images with adaptive detection")
    print("3. Compare extraction results")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Test single image
        input_dir = Path("data/from_jeremy/images_aligned_to_first")
        if not input_dir.exists():
            print(f" Directory not found: {input_dir}")
            exit()
        
        # Get first few images
        test_images = sorted(list(input_dir.glob("*.jpg")))[:3]
        
        print(f"\nAvailable test images:")
        for i, img_path in enumerate(test_images):
            print(f"  {i+1}. {img_path.name}")
        
        img_choice = input("\nEnter image number to test (1-3): ").strip()
        try:
            idx = int(img_choice) - 1
            if 0 <= idx < len(test_images):
                test_single_image_detection(test_images[idx])
            else:
                print("Invalid choice")
        except:
            print("Invalid choice")
    
    elif choice == '2':
        # Batch extract all images
        batch_adaptive_extraction()
    
    elif choice == '3':
        # Compare extraction methods
        print("\n Comparing extraction methods...")
        print("1. Adaptive detection (this script)")
        print("2. Fixed coordinates (previous method)")
        print("3. Both")
        
        comp_choice = input("Enter choice (1-3): ").strip()
        
        if comp_choice in ['1', '3']:
            print("\nRunning adaptive extraction...")
            batch_adaptive_extraction()
        
        if comp_choice in ['2', '3']:
            print("\nRunning fixed coordinate extraction...")
            # You would run your previous extraction script here
            print("To run fixed coordinate extraction, use: python scripts/batch_smart_extraction.py")
    
    else:
        print("Invalid choice")