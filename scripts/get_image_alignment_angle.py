# scripts/straight_as_first.py
"""
MAKE ALL IMAGES AS STRAIGHT AS THE FIRST IMAGE!
Uses m-t0627-00538-00634.jpg as reference, aligns all others to it.
"""

import cv2
import numpy as np
from pathlib import Path
import math

def get_image_alignment_angle(image_path):
    """
    Get the angle needed to make image straight.
    Returns: rotation_angle, detected_raw_angle
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect vertical lines (form columns)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find vertical lines specifically
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0.0, 0.0
    
    vertical_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Vertical lines should be near 90° or -90°
        if abs(angle) > 80:  # Close to vertical
            # Deviation from perfect vertical (90°)
            deviation = 90 - abs(angle)
            # Add sign based on direction
            vertical_angles.append(deviation * (1 if angle > 0 else -1))
    
    if vertical_angles:
        median_angle = np.median(vertical_angles)
        return median_angle, median_angle
    
    return 0.0, 0.0

def align_to_reference(reference_image_path, target_image_path):
    """
    Make target image match the reference image's alignment.
    """
    # Get reference angle (should be close to 0 if reference is straight)
    ref_angle, _ = get_image_alignment_angle(reference_image_path)
    
    # Get target angle
    target_angle, raw_angle = get_image_alignment_angle(target_image_path)
    
    # Calculate correction needed
    # If ref is straight (ref_angle ≈ 0), correction = -target_angle
    # But let's be more robust:
    correction_needed = -target_angle  # Simple: negate target's tilt
    
    # Also account for reference if it's not perfectly 0
    correction_needed += ref_angle
    
    return correction_needed, raw_angle, ref_angle

def batch_align_to_first():
    """
    Align ALL images to the first image (m-t0627-00538-00634.jpg)
    """
    input_dir = Path("data/Research Final Ver. Jeremy P")
    output_dir = Path("data/from_jeremy/images_aligned_to_first")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images sorted
    all_images = sorted(list(input_dir.glob("*.jpg")))
    
    if not all_images:
        print("❌ No images found!")
        return
    
    # Reference image (first one, should be straight)
    reference_path = all_images[0]
    print(f"🎯 REFERENCE IMAGE: {reference_path.name}")
    
    # Get reference alignment
    ref_angle, _ = get_image_alignment_angle(reference_path)
    print(f"   Reference angle: {ref_angle:.2f}° (should be close to 0)")
    
    # Process all images
    print(f"\n🔄 Aligning {len(all_images)} images to reference...")
    print("="*60)
    
    rotation_stats = []
    
    for i, img_path in enumerate(all_images):
        print(f"[{i+1}/{len(all_images)}] {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Could not load, skipping")
            continue
        
        if img_path == reference_path:
            # Copy reference as-is
            cv2.imwrite(str(output_dir / img_path.name), img)
            rotation_stats.append((img_path.name, 0.0, 0.0, ref_angle))
            print(f"  ✓ Reference image, copied as-is")
            continue
        
        # Calculate alignment correction
        correction_needed, raw_angle, ref_angle_val = align_to_reference(reference_path, img_path)
        
        # Apply correction if significant
        if abs(correction_needed) > 0.3:
            print(f"  📐 Raw: {raw_angle:.2f}°, Correction: {correction_needed:.2f}°")
            
            # Rotate image
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, correction_needed, 1.0)
            
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            
            new_width = int(height * sin_val + width * cos_val)
            new_height = int(height * cos_val + width * sin_val)
            
            rotation_matrix[0, 2] += (new_width - width) / 2
            rotation_matrix[1, 2] += (new_height - height) / 2
            
            rotated = cv2.warpAffine(
                img, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            cv2.imwrite(str(output_dir / img_path.name), rotated)
            rotation_stats.append((img_path.name, raw_angle, correction_needed, ref_angle_val))
            
            # Create visual comparison for debugging - FIXED RESIZING
            if i < 10:  # Only for first 10
                debug_dir = output_dir / "debug_comparisons"
                debug_dir.mkdir(exist_ok=True)
                
                # Resize for comparison - ensure both have same height
                target_height = 600
                
                # Resize original
                orig_height, orig_width = img.shape[:2]
                scale_orig = target_height / float(orig_height)
                new_width_orig = int(orig_width * scale_orig)
                orig_resized = cv2.resize(img, (new_width_orig, target_height))
                
                # Resize rotated
                rot_height, rot_width = rotated.shape[:2]
                scale_rot = target_height / float(rot_height)
                new_width_rot = int(rot_width * scale_rot)
                rot_resized = cv2.resize(rotated, (new_width_rot, target_height))
                
                # Now they have the same height (target_height)
                combined = np.hstack([orig_resized, rot_resized])
                
                cv2.putText(combined, "BEFORE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(combined, f"AFTER ({correction_needed:+.1f}°)",
                    (new_width_orig + 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                debug_path = debug_dir / f"{img_path.stem}_compare.jpg"
                cv2.imwrite(str(debug_path), combined)
        else:
            # Copy as-is (already aligned)
            cv2.imwrite(str(output_dir / img_path.name), img)
            rotation_stats.append((img_path.name, raw_angle, 0.0, ref_angle_val))
            print(f"  ✓ Already aligned")
    
    # Print summary
    print(f"\n" + "="*60)
    print("📊 ALIGNMENT SUMMARY")
    print("="*60)
    
    rotated_count = sum(1 for _, _, correction, _ in rotation_stats if abs(correction) > 0.1)
    
    print(f"Reference: {reference_path.name} ({ref_angle:.2f}°)")
    print(f"Total images: {len(rotation_stats)}")
    print(f"Images rotated: {rotated_count}")
    print(f"Images already aligned: {len(rotation_stats) - rotated_count}")
    
    # Show problematic images (large corrections)
    print(f"\n⚠️  LARGEST CORRECTIONS:")
    rotation_stats.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for name, raw, correction, _ in rotation_stats[:10]:
        if abs(correction) > 0.5:
            print(f"  {name}: raw={raw:.2f}°, correction={correction:+.2f}°")
    
    # Save detailed log
    log_path = output_dir / "alignment_log.txt"
    with open(log_path, 'w') as f:
        f.write("IMAGE ALIGNMENT LOG\n")
        f.write("="*50 + "\n")
        f.write(f"Reference: {reference_path.name} (angle: {ref_angle:.2f}°)\n")
        f.write(f"Total images: {len(rotation_stats)}\n")
        f.write(f"Images rotated: {rotated_count}\n\n")
        
        f.write("DETAILS:\n")
        f.write("Image, Raw Angle, Correction Applied\n")
        f.write("-"*50 + "\n")
        
        for name, raw, correction, _ in rotation_stats:
            f.write(f"{name}, {raw:.2f}, {correction:.2f}\n")
    
    print(f"\n✅ Alignment complete!")
    print(f"   Aligned images: {output_dir}")
    print(f"   Debug comparisons: {output_dir}/debug_comparisons/")
    print(f"   Log file: {log_path}")
    
    return output_dir

def verify_extraction_on_aligned():
    """
    Quick test: Try extracting cells from aligned images to verify.
    """
    aligned_dir = Path("data/from_jeremy/images_aligned_to_first")
    
    if not aligned_dir.exists():
        print("❌ Aligned images not found! Run alignment first.")
        return
    
    # Pick first few images
    test_images = sorted(list(aligned_dir.glob("*.jpg")))[:3]
    
    print("🧪 VERIFYING EXTRACTION ON ALIGNED IMAGES")
    print("="*60)
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check head column area (column 5: 1889-2204)
        first_row_y = 1263
        row_height = 78
        
        head_cells = []
        for row in range(3):  # Check first 3 rows
            y_start = first_row_y + row * row_height
            y_end = y_start + row_height
            
            head_cell = gray[y_start:y_end, 1889:2204]
            
            if head_cell.size > 0:
                # Check if it has writing (dark pixels)
                darkness = np.sum(head_cell < 128) / head_cell.size
                has_writing = darkness > 0.05
                
                head_cells.append((row, has_writing, darkness))
        
        print(f"  Head column check (rows 0-2):")
        for row, has_writing, darkness in head_cells:
            status = "✓" if has_writing else "✗"
            print(f"    Row {row}: {status} (darkness: {darkness:.1%})")
        
        # Visual check: save sample head cell
        if head_cells:
            sample_row = 0
            y_start = first_row_y + sample_row * row_height
            y_end = y_start + row_height
            
            sample_cell = img[y_start:y_end, 1889:2204]
            
            debug_dir = Path("extraction_verification")
            debug_dir.mkdir(exist_ok=True)
            
            sample_path = debug_dir / f"{img_path.stem}_head_cell_row{sample_row}.jpg"
            cv2.imwrite(str(sample_path), sample_cell)
            
            print(f"  Sample head cell saved: {sample_path}")

if __name__ == "__main__":
    print("="*70)
    print("🔥 ALIGN ALL IMAGES TO FIRST (STRAIGHT) IMAGE")
    print("="*70)
    
    print("\nThis will:")
    print("1. Use m-t0627-00538-00634.jpg as reference (should be straight)")
    print("2. Measure how each image differs from the reference")
    print("3. Rotate each image to match the reference alignment")
    print("4. Save aligned images ready for extraction")
    
    confirm = input("\nAlign ALL images to first image? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        output_dir = batch_align_to_first()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Check debug comparisons: {output_dir}/debug_comparisons/")
        print(f"2. Verify alignment looks good")
        print(f"3. Update extraction script to use: {output_dir}/")
        print(f"4. Run extraction: python scripts/batch_smart_extraction.py")
        
        print(f"\n🧪 Quick extraction test:")
        verify_extraction_on_aligned()
        
    else:
        print("Alignment cancelled.")