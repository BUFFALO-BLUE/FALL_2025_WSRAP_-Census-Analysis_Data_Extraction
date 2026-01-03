# scripts/shape_based_zero_detector.py
"""
SHAPE-BASED '0' DETECTOR
Uses contour analysis and shape matching to find circular '0' shapes.
"""


# Fix Windows console encoding for emojis
import sys
import io

# Store original print function
_original_print = print

def safe_print(*args, **kwargs):
    """Print function that handles encoding errors gracefully"""
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: replace emojis with ASCII equivalents
        text = ' '.join(str(arg) for arg in args)
        # Replace common emojis with ASCII
        replacements = {
            '🎯': '[TARGET]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📸': '[CAMERA]',
            '🔍': '[SEARCH]',
            '🎛️': '[CALIBRATE]',
            '📊': '[CHART]',
            '💾': '[SAVE]',
            '⚠️': '[WARNING]',
            '🎉': '[SUCCESS]'
        }
        for emoji, replacement in replacements.items():
            text = text.replace(emoji, replacement)
        _original_print(text, **kwargs)

if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, TypeError):
        # If UTF-8 setup fails, use safe_print as fallback
        pass
    
    # Always use safe_print on Windows as backup (handles cases where UTF-8 still fails)
    # We'll test if UTF-8 works, and if not, safe_print will catch encoding errors
    try:
        # Test if we can print emoji
        test_output = io.StringIO()
        test_wrapper = io.TextIOWrapper(test_output, encoding='utf-8', errors='replace')
        test_wrapper.write('🎯')
        test_wrapper.flush()
    except:
        # If test fails, override print with safe_print
        print = safe_print

# #region agent log
import json
import os
log_path = r"c:\Users\Musarah\Downloads\FALL_2025_WSRAP_-Census-Analysis_Data_Extraction\.cursor\debug.log"
try:
    import subprocess
    import platform
    # Capture invocation details
    invocation_data = {
        "script": "accurate_head_detector.py",
        "platform": sys.platform,
        "sys_executable": sys.executable if hasattr(sys, 'executable') else "N/A",
        "sys_argv": sys.argv if hasattr(sys, 'argv') else [],
        "cwd": os.getcwd(),
        "script_path": __file__ if '__file__' in globals() else "N/A",
        "python_version": sys.version.split()[0] if hasattr(sys, 'version') else "N/A",
        "path_env": os.environ.get('PATH', '')[:200] if 'PATH' in os.environ else "N/A",  # First 200 chars
        "shell": os.environ.get('SHELL', os.environ.get('COMSPEC', 'N/A')),
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"accurate_head_detector.py:64","message":"Script started - invocation details","data":invocation_data,"timestamp":int(__import__('time').time()*1000)}) + "\n")
except Exception as e:
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"accurate_head_detector.py:64","message":"Script start logging failed","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
# #endregion

import cv2
import numpy as np
import json as json_module
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# #region agent log
try:
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"accurate_head_detector.py:25","message":"Imports completed successfully","data":{"cv2_loaded":hasattr(cv2,'imread'),"numpy_loaded":hasattr(np,'array'),"pathlib_loaded":hasattr(Path,'__class__')},"timestamp":int(__import__('time').time()*1000)}) + "\n")
except: pass
# #endregion

# Smart adaptive row detection functions (from smart_adaptive_extraction.py)
def find_smart_row_boundaries(image, start_y, expected_height, target_rows):
    """Find row boundaries that adapt to each row"""
    height, width = image.shape
    boundaries = [start_y]
    current_y = start_y
    
    for row_num in range(target_rows):
        search_start = current_y
        search_end = min(current_y + expected_height * 2, height - 1)
        
        if search_start >= height:
            break
            
        optimal_bottom = find_optimal_row_bottom_improved(image, search_start, search_end, expected_height)
        
        if optimal_bottom is None:
            optimal_bottom = current_y + expected_height
        
        gap = optimal_bottom - current_y
        min_gap = expected_height * 0.6
        max_gap = expected_height * 1.4
        
        if gap < min_gap or gap > max_gap:
            optimal_bottom = current_y + expected_height
        
        boundaries.append(optimal_bottom)
        current_y = optimal_bottom
    
    return boundaries

def find_optimal_row_bottom_improved(image, start_y, end_y, expected_height):
    """Improved method to find the best bottom boundary for a row"""
    search_region = image[start_y:end_y, :]
    if search_region.size == 0:
        return None
    
    horizontal_proj = np.sum(search_region == 0, axis=1)
    
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_proj = np.convolve(horizontal_proj, kernel, mode='same')
    
    search_center = expected_height
    search_window = 30
    
    search_start = max(0, search_center - search_window)
    search_end = min(len(smoothed_proj), search_center + search_window)
    
    if search_end > search_start:
        window_proj = smoothed_proj[search_start:search_end]
        minima_positions = []
        for i in range(1, len(window_proj) - 1):
            if window_proj[i] < window_proj[i-1] and window_proj[i] < window_proj[i+1]:
                minima_positions.append(i)
        
        if minima_positions:
            min_values = [window_proj[pos] for pos in minima_positions]
            deepest_min_idx = minima_positions[np.argmin(min_values)]
            optimal_pos = search_start + deepest_min_idx
        else:
            min_pos = np.argmin(window_proj)
            optimal_pos = search_start + min_pos
    else:
        optimal_pos = expected_height
    
    return start_y + optimal_pos

class ShapeZeroDetector:
    def __init__(self, head_x1=1889, head_x2=2204, debug=False):
        self.head_x1 = head_x1
        self.head_x2 = head_x2
        self.debug = debug
        if debug:
            self.debug_dir = Path("data/debug_shape_detection")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_zero_by_shape(self, head_cell):
        """
        Detect if a cell contains a '0' using shape analysis.
        Returns: True if '0' is detected, False otherwise
        """
        if head_cell.size == 0:
            return False
        
        height, width = head_cell.shape
        
        # Step 1: Preprocess - invert and threshold
        _, binary = cv2.threshold(head_cell, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Step 2: Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Step 3: Find the largest contour (likely the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Skip if too small
        if area < (height * width * 0.02):  # Less than 2% of cell area
            return False
        
        # Step 4: Calculate circularity - '0' should be circular
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # '0' is circular (0.7-1.3), other digits are less circular
        is_circular = 0.7 <= circularity <= 1.3
        
        # Step 5: Check for hole in the middle (characteristic of '0')
        # Create mask from contour
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Find internal contours (holes)
        hole_contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        has_hole = len(hole_contours) >= 2  # At least 2 contours = outer + hole
        
        # Step 6: Aspect ratio - '0' is roughly as tall as wide
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = h / w if w > 0 else 0
        
        is_roundish = 0.7 <= aspect_ratio <= 1.5  # Not too tall, not too wide
        
        # Step 7: Solidity - '0' should have a hole, so lower solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        has_low_solidity = solidity < 0.9  # '0' has hole, so solidity < 1
        
        # Step 8: Check if contour is centered in cell
        cell_center_x, cell_center_y = width // 2, height // 2
        contour_center_x = x + w // 2
        contour_center_y = y + h // 2
        
        # Allow 30% deviation from center
        is_centered = (abs(contour_center_x - cell_center_x) < width * 0.3 and 
                      abs(contour_center_y - cell_center_y) < height * 0.3)
        
        # DEBUG: Save analysis
        if self.debug:
            self.save_debug_info(head_cell, binary, largest_contour, 
                                circularity, has_hole, aspect_ratio, 
                                solidity, is_centered)
        
        # FINAL DECISION: Must meet multiple criteria
        criteria_met = 0
        total_criteria = 5
        
        if is_circular:
            criteria_met += 1
        if has_hole:
            criteria_met += 1
        if is_roundish:
            criteria_met += 1
        if has_low_solidity:
            criteria_met += 1
        if is_centered:
            criteria_met += 1
        
        # At least 3 out of 5 criteria must be met
        return criteria_met >= 3
    
    def save_debug_info(self, original, binary, contour, 
                       circularity, has_hole, aspect_ratio, 
                       solidity, is_centered):
        """Save debug information for analysis."""
        # Create visualization
        viz = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Draw contour
        cv2.drawContours(viz, [contour], -1, (0, 255, 0), 2)
        
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(viz, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw convex hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(viz, [hull], -1, (0, 0, 255), 2)
        
        # Add text with metrics
        metrics = [
            f"Circularity: {circularity:.2f}",
            f"Hole: {'YES' if has_hole else 'NO'}",
            f"Aspect: {aspect_ratio:.2f}",
            f"Solidity: {solidity:.2f}",
            f"Centered: {'YES' if is_centered else 'NO'}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(viz, metric, (10, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save images
        filename = f"circ{circularity:.2f}_hole{has_hole}_asp{aspect_ratio:.2f}"
        cv2.imwrite(str(self.debug_dir / f"{filename}_original.png"), original)
        cv2.imwrite(str(self.debug_dir / f"{filename}_binary.png"), binary)
        cv2.imwrite(str(self.debug_dir / f"{filename}_analysis.png"), viz)
    
    def detect_head_rows(self, image_path):
        """
        Detect head rows in an image using shape-based '0' detection with smart adaptive row boundaries.
        """
        img = cv2.imread(str(image_path), 0)
        if img is None:
            return []
        
        # Use smart adaptive row detection
        first_row_y = 1263
        expected_row_height = 78
        target_rows = 40
        
        # Get adaptive row boundaries
        row_boundaries = find_smart_row_boundaries(img, first_row_y, expected_row_height, target_rows)
        
        head_rows = []
        
        for row_idx in range(len(row_boundaries) - 1):
            y1 = row_boundaries[row_idx]
            y2 = row_boundaries[row_idx + 1]
            
            if y2 > img.shape[0] or y1 >= img.shape[0]:
                break
            
            head_cell = img[y1:y2, self.head_x1:self.head_x2]
            
            if head_cell.size > 0:
                # Check if this cell contains a '0'
                if self.detect_zero_by_shape(head_cell):
                    head_rows.append(row_idx)
        
        return head_rows

def test_shape_detection():
    """Test shape detection on sample images."""
    
    print("🧪 TESTING SHAPE-BASED '0' DETECTION")
    print("="*60)
    
    # Load sample image
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    image_paths = sorted(list(input_dir.glob("*.jpg")))[:2]
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    detector = ShapeZeroDetector(debug=True)
    
    for img_path in image_paths:
        print(f"\nTesting: {img_path.name}")
        
        img = cv2.imread(str(img_path), 0)
        
        # Test first 5 rows
        first_row_y = 1263
        row_height = 78
        head_x1, head_x2 = 2150, 2200
        
        results = []
        
        for row in range(5):
            y1 = first_row_y + (row * row_height)
            y2 = y1 + row_height
            
            head_cell = img[y1:y2, head_x1:head_x2]
            
            if head_cell.size > 0:
                is_zero = detector.detect_zero_by_shape(head_cell)
                results.append((row, is_zero))
                
                print(f"  Row {row}: {'0' if is_zero else 'not 0'}")
        
        print(f"\n  Summary: {sum(1 for _, is_zero in results if is_zero)}/{len(results)} rows contain '0'")

def create_training_samples():
    """Create training samples for manual verification."""
    
    print("📸 CREATING TRAINING SAMPLES FOR MANUAL VERIFICATION")
    print("="*60)
    
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    image_paths = sorted(list(input_dir.glob("*.jpg")))[:5]  # First 5 images
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    detector = ShapeZeroDetector()
    
    samples_dir = Path("data/shape_verification_samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    for img_idx, img_path in enumerate(image_paths):
        print(f"\nProcessing: {img_path.name}")
        
        img = cv2.imread(str(img_path), 0)
        
        # Use smart adaptive row detection
        first_row_y = 1263
        expected_row_height = 78
        target_rows = 40
        head_x1, head_x2 = 1889, 2204  # Updated to match smart_adaptive
        
        row_boundaries = find_smart_row_boundaries(img, first_row_y, expected_row_height, target_rows)
        
        for row_idx in range(min(10, len(row_boundaries) - 1)):  # First 10 rows
            y1 = row_boundaries[row_idx]
            y2 = row_boundaries[row_idx + 1]
            
            head_cell = img[y1:y2, head_x1:head_x2]
            
            if head_cell.size > 0:
                # Save the cell
                filename = f"img{img_idx}_row{row_idx:02d}.png"
                filepath = samples_dir / filename
                cv2.imwrite(str(filepath), head_cell)
                
                # Detect if it's '0'
                is_zero = detector.detect_zero_by_shape(head_cell)
                
                all_samples.append({
                    'filename': filename,
                    'image': img_path.name,
                    'row': row_idx,
                    'is_zero': is_zero,
                    'prediction': '0' if is_zero else 'other'
                })
    
    # Save sample information
    samples_info = samples_dir / "samples_info.csv"
    with open(samples_info, 'w') as f:
        f.write("filename,image,row,is_zero,prediction\n")
        for sample in all_samples:
            f.write(f"{sample['filename']},{sample['image']},{sample['row']},"
                   f"{sample['is_zero']},{sample['prediction']}\n")
    
    print(f"\n✅ Created {len(all_samples)} samples in: {samples_dir}")
    print(f"   Sample info: {samples_info}")
    print(f"\n🎯 MANUAL VERIFICATION NEEDED:")
    print(f"   Please check the samples and verify if '0' detection is correct.")
    print(f"   Update the CSV file with correct labels if needed.")

def batch_shape_detection():
    """Run shape-based detection on all images."""
    
    print("="*80)
    print("🎯 SHAPE-BASED '0' DETECTION - ALL IMAGES")
    print("="*80)
    
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    image_paths = sorted(list(input_dir.glob("*.jpg")))
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    detector = ShapeZeroDetector(debug=False)
    results = {}
    
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {img_path.name}")
        
        try:
            head_rows = detector.detect_head_rows(img_path)
            results[img_path.name] = {
                'head_rows': head_rows,
                'num_head_rows': len(head_rows),
                'success': True
            }
            print(f"  Found {len(head_rows)} head rows: {head_rows}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[img_path.name] = {
                'head_rows': [],
                'num_head_rows': 0,
                'success': False,
                'error': str(e)
            }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"data/shape_based_detection_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_summary_shape(results, output_file)
    
    return results, output_file

def print_summary_shape(results, output_file):
    """Print summary for shape-based detection."""
    
    print(f"\n{'='*80}")
    print("📊 SHAPE-BASED DETECTION SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total_images = len(results)
    
    head_row_counts = [r['num_head_rows'] for r in results.values() if r['success']]
    
    if head_row_counts:
        avg_head_rows = np.mean(head_row_counts)
        std_head_rows = np.std(head_row_counts)
        min_head_rows = min(head_row_counts)
        max_head_rows = max(head_row_counts)
        
        print(f"Images processed: {total_images}")
        print(f"Successful detections: {successful}")
        print(f"Average head rows per image: {avg_head_rows:.1f}")
        print(f"Standard deviation: {std_head_rows:.1f}")
        print(f"Minimum head rows: {min_head_rows}")
        print(f"Maximum head rows: {max_head_rows}")
        
        # Distribution
        print(f"\n📈 DISTRIBUTION OF HEAD ROWS:")
        counts = {}
        for count in head_row_counts:
            counts[count] = counts.get(count, 0) + 1
        
        for count in sorted(counts.keys()):
            percentage = (counts[count] / successful) * 100
            print(f"  {count:2d} head rows: {counts[count]:3d} images ({percentage:.1f}%)")
        
        # Check if reasonable
        if 5 <= avg_head_rows <= 20:
            print(f"\n✅ REASONABLE: {avg_head_rows:.1f} head rows per image (expected: 5-20)")
        else:
            print(f"\n⚠️  UNUSUAL: {avg_head_rows:.1f} head rows per image (expected: 5-20)")
    
    print(f"\n💾 Results saved to: {output_file}")

def compare_methods():
    """Compare different detection methods."""
    
    print("🔍 COMPARING DETECTION METHODS")
    print("="*60)
    
    # Test on a single image
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    image_paths = sorted(list(input_dir.glob("*.jpg")))[:1]
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    test_image = image_paths[0]
    print(f"Testing on: {test_image.name}")
    
    img = cv2.imread(str(test_image), 0)
    
    # Use smart adaptive row detection
    first_row_y = 1263
    expected_row_height = 78
    target_rows = 40
    head_x1, head_x2 = 1889, 2204  # Updated to match smart_adaptive
    
    row_boundaries = find_smart_row_boundaries(img, first_row_y, expected_row_height, target_rows)
    
    # Test different rows
    test_rows = [0, 1, 2, 3, 4, 5]
    
    print(f"\nRow | Pixel% | Shape | Manual Check")
    print("-"*40)
    
    for row_idx in test_rows:
        if row_idx >= len(row_boundaries) - 1:
            continue
        
        y1 = row_boundaries[row_idx]
        y2 = row_boundaries[row_idx + 1]
        
        head_cell = img[y1:y2, head_x1:head_x2]
        
        if head_cell.size > 0:
            # Pixel-based method
            _, binary = cv2.threshold(head_cell, 180, 255, cv2.THRESH_BINARY_INV)
            black_pixels = np.sum(binary == 255)
            total_pixels = head_cell.shape[0] * head_cell.shape[1]
            black_percentage = black_pixels / total_pixels if total_pixels > 0 else 0
            
            # Shape-based method
            detector = ShapeZeroDetector()
            is_zero_shape = detector.detect_zero_by_shape(head_cell)
            
            # Save for manual checking
            sample_dir = Path("data/method_comparison")
            sample_dir.mkdir(exist_ok=True)
            
            filename = f"row{row_idx:02d}_pixel{black_percentage:.2f}_shape{is_zero_shape}.png"
            cv2.imwrite(str(sample_dir / filename), head_cell)
            
            print(f"{row_idx:3d} | {black_percentage:.2f}    | {'0' if is_zero_shape else 'X'}     | {filename}")

def manual_calibration():
    """
    Manual calibration tool - helps find the right parameters.
    """
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:444","message":"manual_calibration function entered","data":{},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    print("🎛️ MANUAL CALIBRATION TOOL")
    print("="*60)
    print("This helps you manually calibrate the '0' detection parameters.")
    print()
    
    # Create calibration samples
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:456","message":"Before glob - checking input_dir","data":{"input_dir":str(input_dir),"exists":input_dir.exists()},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    image_paths = sorted(list(input_dir.glob("*.jpg")))[:3]
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:460","message":"After glob - image paths found","data":{"num_paths":len(image_paths),"paths":[str(p) for p in image_paths[:3]]},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    calibration_dir = Path("data/calibration_samples")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:464","message":"calibration_dir created","data":{"calibration_dir":str(calibration_dir),"exists":calibration_dir.exists()},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    print("Creating calibration samples...")
    
    sample_count = 0
    calibration_data = []
    
    for img_idx, img_path in enumerate(image_paths):
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:467","message":"Processing image","data":{"img_idx":img_idx,"img_path":str(img_path),"img_path_type":type(img_path).__name__},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except: pass
        # #endregion
        
        img = cv2.imread(str(img_path), 0)
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"accurate_head_detector.py:472","message":"After cv2.imread","data":{"img_loaded":img is not None,"img_shape":img.shape if img is not None else None,"img_path_str":str(img_path)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except: pass
        # #endregion
        
        # Use smart adaptive row detection (same as smart_adaptive_grid)
        first_row_y = 1263
        expected_row_height = 78
        target_rows = 40
        
        # Get adaptive row boundaries using smart method
        row_boundaries = find_smart_row_boundaries(img, first_row_y, expected_row_height, target_rows)
        
        # Head column coordinates (from smart_adaptive - matches smart_adaptive_grid.png)
        head_x1, head_x2 = 1889, 2204
        
        # Take samples from various rows
        sample_rows = [0, 5, 10, 15, 20, 25, 30, 35, min(39, len(row_boundaries)-2)]
        
        for row in sample_rows:
            if row >= len(row_boundaries) - 1:
                continue
            
            y1 = row_boundaries[row]
            y2 = row_boundaries[row + 1]
            
            if y2 > img.shape[0] or y1 >= img.shape[0]:
                continue
            
            head_cell = img[y1:y2, head_x1:head_x2]
            
            if head_cell.size > 0:
                # Save sample
                filename = f"sample_{sample_count:03d}_img{img_idx}_row{row}.png"
                filepath = calibration_dir / filename
                
                # #region agent log
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"accurate_head_detector.py:495","message":"Before cv2.imwrite","data":{"filepath":str(filepath),"filepath_type":type(filepath).__name__,"filepath_parent":str(filepath.parent) if hasattr(filepath,'parent') else "N/A"},"timestamp":int(__import__('time').time()*1000)}) + "\n")
                except: pass
                # #endregion
                
                cv2.imwrite(str(filepath), head_cell)
                
                # #region agent log
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"accurate_head_detector.py:502","message":"After cv2.imwrite","data":{"filepath":str(filepath),"file_exists":filepath.exists()},"timestamp":int(__import__('time').time()*1000)}) + "\n")
                except: pass
                # #endregion
                
                calibration_data.append({
                    'filename': filename,
                    'image': img_path.name,
                    'row': row,
                    'filepath': str(filepath)
                })
                
                sample_count += 1
    
    print(f"\n✅ Created {sample_count} calibration samples in: {calibration_dir}")
    
    # Create calibration guide
    guide = f"""CALIBRATION GUIDE
================

You have {sample_count} sample images in: {calibration_dir}

STEP 1: MANUAL LABELING
----------------------
Look at each PNG file and determine if it contains:
1. '0' (head of household) - CIRCLE WITH HOLE
2. '1', '2', '3' (family members) - NOT CIRCULAR
3. Empty/blank - NO WRITING

STEP 2: CREATE LABELS CSV
------------------------
Create a file called 'labels.csv' in the same folder with format:
filename,is_zero,notes
sample_000_img0_row0.png,TRUE,clear circle
sample_001_img0_row5.png,FALSE,number 1
...

STEP 3: TRAIN DETECTOR
---------------------
Once you have labels, we can train a better detector.

STEP 4: TEST PARAMETERS
----------------------
We'll adjust these parameters based on your labels:
- Circularity threshold (0.7-1.3)
- Aspect ratio (0.7-1.5)
- Minimum area (2% of cell)
- Solidity (hole detection)

NEXT STEPS:
1. Manually label 20-30 samples
2. Save as labels.csv
3. Run calibration analysis
4. Update detection parameters
"""
    
    guide_path = calibration_dir / "CALIBRATION_GUIDE.txt"
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"\n📖 Calibration guide saved: {guide_path}")
    print(f"\n🎯 NEXT: Manually label samples, then we'll optimize the detector.")

if __name__ == "__main__":
    # #region agent log
    try:
        import sys
        import os
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"accurate_head_detector.py:550","message":"Main block entered","data":{"sys_executable":sys.executable if hasattr(sys,'executable') else "N/A","cwd":os.getcwd(),"script_path":__file__ if '__file__' in globals() else "N/A"},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except Exception as e:
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"accurate_head_detector.py:550","message":"Main block entry logging failed","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except: pass
    # #endregion
    
    print("="*80)
    print("🎯 SHAPE-BASED '0' DETECTION - CIRCLE FINDING")
    print("="*80)
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"accurate_head_detector.py:565","message":"Before input prompt","data":{},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    print("\nChoose action:")
    print("1. Test shape detection on sample images")
    print("2. Create training samples for manual verification")
    print("3. Run shape-based detection on ALL images")
    print("4. Compare different detection methods")
    print("5. Manual calibration tool (RECOMMENDED for accuracy)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"accurate_head_detector.py:575","message":"After input - choice received","data":{"choice":choice},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    
    if choice == "1":
        test_shape_detection()
    elif choice == "2":
        create_training_samples()
    elif choice == "3":
        batch_shape_detection()
    elif choice == "4":
        compare_methods()
    elif choice == "5":
        manual_calibration()
    else:
        print("Running recommended option 5...")
        manual_calibration()