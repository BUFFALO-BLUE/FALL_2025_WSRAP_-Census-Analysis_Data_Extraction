import cv2
import numpy as np
import os

def smart_adaptive_extraction():
    """Smart adaptive extraction with ALL required columns and head filtering"""
    
    # Load the preprocessed image
    original = cv2.imread('data/processed/preprocessed_image.png', 0)
    if original is None:
        print("Error: Could not load preprocessed image")
        return
    
    print("=== SMART ADAPTIVE EXTRACTION ===")
    print(f"Image size: {original.shape}")
    
    # ALL REQUIRED COLUMNS FROM OUR MEETING
    columns = {
        # Column 1 Name of Steer
        'street': (629, 718),
        # Column 2 - House number
        'house_number': (718, 836),
        
        # Column 4 - Rented or owned
        'rented_owned': (914, 994),
        
        # Column 5 - Price of rent 
        'price_rent': (996, 1143),
        
        # Column 8 - Head indicator (look for 'o')
        'head': (1889, 2204),
        
        # Column 9 - Gender 
        'gender': (2204, 2285),

        # Column 10 - Race
        'race': (2285, 2388),
        
        # Column 12 - Marital status (FIXED coordinates)
        'marital_status': (2491, 2574),
        
        # Column 26 - Hours worked
        'hours_worked': (4939, 5092),
        
        # Column 32 - Wages
        'wages': (6433, 6588)
    }
    
    # YOUR EXACT STARTING POINT
    first_row_y = 1263
    expected_row_height = 78
    num_rows = 40
    
    print("Using your exact parameters:")
    print(f"First row: y={first_row_y}")
    print(f"Expected row height: {expected_row_height}px")
    print(f"Target: {num_rows} rows")
    
    print("\nColumns being extracted:")
    for col_name, (x1, x2) in columns.items():
        print(f"  {col_name}: x={x1} to x={x2} (width: {x2-x1}px)")
    
    # Step 1: Find ACTUAL row boundaries using improved detection
    print("\nFinding actual row boundaries with improved adaptive detection...")
    row_boundaries = find_smart_row_boundaries(original, first_row_y, expected_row_height, num_rows)
    
    print(f"Found {len(row_boundaries) - 1} rows")
    
    # Step 2: Extract cells with head filtering
    cells = []
    head_rows_indices = []
    os.makedirs('data/extracted_cells', exist_ok=True)
    
    # Create separate folders for head vs non-head rows
    head_output_dir = 'data/extracted_cells/head_rows'
    non_head_output_dir = 'data/extracted_cells/non_head_rows'
    os.makedirs(head_output_dir, exist_ok=True)
    os.makedirs(non_head_output_dir, exist_ok=True)
    
    for row_idx in range(len(row_boundaries) - 1):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        actual_height = y2 - y1
        
        # Extract head column to check for 'o'
        head_cell_img = original[y1:y2, 1889:2204]
        
        # Detect head indicator
        is_head = detect_head_indicator(head_cell_img)
        
        # Choose output directory based on head status
        output_dir = head_output_dir if is_head else non_head_output_dir
        
        if is_head:
            head_rows_indices.append(row_idx)
            print(f"Row {row_idx}: HEAD (y={y1}-{y2}, height={actual_height}px)")
        else:
            print(f"Row {row_idx}: not head (y={y1}-{y2}, height={actual_height}px)")
        
        # Extract ALL columns for this row
        for col_name, (x1, x2) in columns.items():
            cell_img = original[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                # Create filename indicating head status
                head_prefix = "HEAD_" if is_head else ""
                filename = f"{head_prefix}row{row_idx:02d}_{col_name}.png"
                cv2.imwrite(f'{output_dir}/{filename}', cell_img)
                
                # For head rows, also add to cells list for visualization
                if is_head:
                    cells.append((x1, y1, x2, y2, row_idx, col_name, actual_height, is_head))
    
    print(f"\n✅ Extraction complete!")
    print(f"Head rows found: {len(head_rows_indices)}")
    print(f"Non-head rows: {len(row_boundaries) - 1 - len(head_rows_indices)}")
    print(f"Total cells extracted from head rows: {len(head_rows_indices) * len(columns)}")
    
    # Create visualization
    create_smart_visualization(original, columns, row_boundaries, cells, head_rows_indices)
    
    # Create summary report
    create_extraction_report(columns, head_rows_indices, row_boundaries)
    
    return len(head_rows_indices), len(columns)

def detect_head_indicator(head_cell_image):
    """Detect if 'o' is present in head column"""
    
    height, width = head_cell_image.shape
    
    # Check if cell has content (not empty)
    black_pixels = np.count_nonzero(head_cell_image == 0)
    total_pixels = height * width
    black_percentage = (black_pixels / total_pixels) * 100
    
    # 'o' typically has moderate black percentage
    # Adjust these thresholds based on your actual data
    if 10 < black_percentage < 40:  # 'o' usually has moderate ink density
        return True
    
    return False

def find_smart_row_boundaries(image, start_y, expected_height, target_rows):
    """Find row boundaries that adapt to each row"""
    
    height, width = image.shape
    boundaries = [start_y]
    current_y = start_y
    
    # Use the entire width for row detection (more robust)
    for row_num in range(target_rows):
        # Calculate search region for this row
        search_start = current_y
        search_end = min(current_y + expected_height * 2, height - 1)
        
        if search_start >= height:
            break
            
        # Look for the optimal row bottom using improved method
        optimal_bottom = find_optimal_row_bottom_improved(image, search_start, search_end, expected_height)
        
        if optimal_bottom is None:
            # Fallback: use expected height
            optimal_bottom = current_y + expected_height
        
        # Ensure reasonable gap between rows
        gap = optimal_bottom - current_y
        min_gap = expected_height * 0.6
        max_gap = expected_height * 1.4
        
        if gap < min_gap:
            optimal_bottom = current_y + expected_height
        elif gap > max_gap:
            optimal_bottom = current_y + expected_height
        
        boundaries.append(optimal_bottom)
        current_y = optimal_bottom
    
    return boundaries

def find_optimal_row_bottom_improved(image, start_y, end_y, expected_height):
    """Improved method to find the best bottom boundary for a row"""
    
    # Extract the search region
    search_region = image[start_y:end_y, :]
    if search_region.size == 0:
        return None
    
    # Calculate horizontal projection
    horizontal_proj = np.sum(search_region == 0, axis=1)
    
    # Smooth the projection to reduce noise
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_proj = np.convolve(horizontal_proj, kernel, mode='same')
    
    # Look for the natural break point (valley in projection)
    # We expect the break to be around the expected_height
    search_center = expected_height
    search_window = 30  # Look ±30 pixels around expected height
    
    search_start = max(0, search_center - search_window)
    search_end = min(len(smoothed_proj), search_center + search_window)
    
    # Find the minimum in this window (the valley between rows)
    if search_end > search_start:
        window_proj = smoothed_proj[search_start:search_end]
        
        # Find all local minima
        minima_positions = []
        for i in range(1, len(window_proj) - 1):
            if window_proj[i] < window_proj[i-1] and window_proj[i] < window_proj[i+1]:
                minima_positions.append(i)
        
        if minima_positions:
            # Choose the deepest minimum (lowest value)
            min_values = [window_proj[pos] for pos in minima_positions]
            deepest_min_idx = minima_positions[np.argmin(min_values)]
            optimal_pos = search_start + deepest_min_idx
        else:
            # Fallback: find global minimum in window
            min_pos = np.argmin(window_proj)
            optimal_pos = search_start + min_pos
    else:
        optimal_pos = expected_height
    
    return start_y + optimal_pos

def create_smart_visualization(image, columns, row_boundaries, cells, head_rows_indices):
    """Create visualization showing the smart grid"""
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw column boundaries (blue)
    for col_name, (x1, x2) in columns.items():
        cv2.line(viz, (x1, 0), (x1, image.shape[0]), (255, 0, 0), 3)
        cv2.line(viz, (x2, 0), (x2, image.shape[0]), (255, 0, 0), 3)
        cv2.putText(viz, col_name, (x1, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw row boundaries (red)
    for i, y in enumerate(row_boundaries):
        cv2.line(viz, (0, y), (image.shape[1], y), (0, 0, 255), 3)
        if i < len(row_boundaries) - 1:
            row_height = row_boundaries[i+1] - y
            # Highlight head rows with different color
            if i in head_rows_indices:
                cv2.putText(viz, f"HEAD Row {i}", (50, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(viz, f"Row {i} ({row_height}px)", (50, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Draw head cells (green highlight)
    for x1, y1, x2, y2, row_idx, col_name, row_height, is_head in cells:
        if is_head:
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.imwrite('data/processed/smart_adaptive_grid.png', viz)
    print("Visualization saved: data/processed/smart_adaptive_grid.png")
    
    # Create zoomed version for head rows
    if head_rows_indices:
        # Zoom on first head row
        first_head_row = head_rows_indices[0]
        y1, y2 = row_boundaries[first_head_row], row_boundaries[first_head_row + 1]
        zoom_region = (0, max(0, y1-100), image.shape[1], min(image.shape[0], y2+100))
        zoomed = viz[zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]]
        cv2.imwrite('data/processed/head_row_zoom.png', zoomed)
        print("Head row zoom saved: data/processed/head_row_zoom.png")

def create_extraction_report(columns, head_rows_indices, row_boundaries):
    """Create a detailed extraction report"""
    
    report = f"""=== SMART ADAPTIVE EXTRACTION REPORT ===

EXTRACTION PARAMETERS:
- First row Y position: {row_boundaries[0]}
- Number of rows extracted: {len(row_boundaries) - 1}
- Head rows detected: {len(head_rows_indices)}

COLUMNS EXTRACTED:
"""
    
    for col_name, (x1, x2) in columns.items():
        report += f"- {col_name}: x={x1}-{x2} (width: {x2-x1}px)\n"
    
    report += f"\nHEAD ROWS INDICES: {head_rows_indices}\n"
    
    report += f"""
OUTPUT STRUCTURE:
- Head rows: data/extracted_cells/head_rows/
- Non-head rows: data/extracted_cells/non_head_rows/
- Visualizations: data/processed/

FILENAMING CONVENTION:
- HEAD_row00_house_number.png
- HEAD_row00_rented_owned.png
- etc.

NEXT STEPS:
1. Verify head detection accuracy
2. Map Excel data to extracted cells
3. Adjust column coordinates if needed
"""
    
    with open('data/processed/smart_extraction_report.txt', 'w') as f:
        f.write(report)
    
    print("Report saved: data/processed/smart_extraction_report.txt")

if __name__ == "__main__":
    head_rows, num_columns = smart_adaptive_extraction()
    print(f"\n🎯 SMART ADAPTIVE EXTRACTION FINISHED!")
    print(f"Extracted {head_rows} head rows × {num_columns} columns")
    print(f"Total: {head_rows * num_columns} cells from head rows")
    print(f"\n📁 Check data/extracted_cells/head_rows/ for the important data")
    print(f"📊 Check data/processed/smart_extraction_report.txt for details")