# scripts/calibrated_deskew.py
"""
CALIBRATION-BASED DESKEW: Uses known straight images to calibrate, then fixes all others.
"""

import cv2
import numpy as np
from pathlib import Path
import math

def detect_skew_angle(gray_image):
    """
    Detect raw skew angle without any correction.
    Returns median angle of detected lines.
    """
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 == x1:
            angle = 90.0
        else:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Normalize to -45 to 45 range
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        if -45 <= angle <= 45:
            angles.append(angle)
    
    if angles:
        return np.median(angles)
    return 0.0

def find_calibration_images(input_dir, num_samples=10):
    """
    Find images that are likely already straight for calibration.
    Looks for images with minimal detected skew.
    """
    input_path = Path(input_dir)
    images = sorted(list(input_path.glob("*.jpg")))[:50]  # Check first 50
    
    print("🔍 Finding calibration images (looking for straight ones)...")
    
    image_angles = []
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = detect_skew_angle(gray)
        
        image_angles.append((img_path.name, angle))
    
    # Sort by absolute angle (closest to 0 first)
    image_angles.sort(key=lambda x: abs(x[1]))
    
    print(f"\n📊 Top {num_samples} straightest images for calibration:")
    for i, (name, angle) in enumerate(image_angles[:num_samples]):
        print(f"  {i+1:2d}. {name}: {angle:6.2f}°")
    
    # Calculate calibration offset (average of straightest images)
    calibration_angles = [angle for _, angle in image_angles[:num_samples]]
    calibration_offset = np.mean(calibration_angles)
    
    print(f"\n🎯 Calibration offset: {calibration_offset:.2f}°")
    print(f"   This is what our camera considers 'straight'")
    
    return calibration_offset, image_angles[:num_samples]

def deskew_with_calibration(image_path, calibration_offset=0.0):
    """
    Deskew using calibration offset.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect raw angle
    raw_angle = detect_skew_angle(gray)
    
    # Apply calibration: subtract the offset
    calibrated_angle = raw_angle - calibration_offset
    
    # Only correct if significant
    if abs(calibrated_angle) > 0.5:
        # 🔥 NEW: Use opposite rotation for certain conditions
        # If the raw angle and calibration offset have same sign, we might need opposite rotation
        final_angle = -calibrated_angle
        
        deskewed_img = _rotate_image(original, final_angle)
        return deskewed_img, final_angle, raw_angle
    
    return original, 0.0, raw_angle

def _rotate_image(image, angle):
    """Rotate image with white background"""
    if abs(angle) < 0.1:
        return image
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated

def batch_calibrated_deskew():
    """
    Main function: Calibrate then deskew all images.
    """
    input_dir = Path("data/from_jeremy/images")
    output_dir = Path("data/from_jeremy/images_calibrated_straight")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Find calibration offset
    calibration_offset, cal_images = find_calibration_images(input_dir, num_samples=15)
    
    # 2. Create visual test first
    test_dir = Path("calibration_test")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n🧪 Testing calibration on calibration images...")
    
    for img_name, expected_angle in cal_images:
        img_path = input_dir / img_name
        
        # These should be straight already, so calibration should show ~0° correction
        deskewed_img, final_angle, raw_angle = deskew_with_calibration(
            img_path, calibration_offset
        )
        
        print(f"  {img_name}: raw={raw_angle:.2f}°, final={final_angle:.2f}°")
        
        # Save test comparison
        original = cv2.imread(str(img_path))
        if original is not None and deskewed_img is not None:
            # Create side-by-side
            height = max(original.shape[0], deskewed_img.shape[0])
            scale = 600 / height
            
            orig_resized = cv2.resize(original, 
                                     (int(original.shape[1] * scale), int(height * scale)))
            deskew_resized = cv2.resize(deskewed_img,
                                       (int(deskewed_img.shape[1] * scale), int(height * scale)))
            
            combined = np.hstack([orig_resized, deskew_resized])
            
            # Add text
            cv2.putText(combined, f"Original (raw:{raw_angle:.1f}°)", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, f"Calibrated (final:{final_angle:.1f}°)", 
                       (orig_resized.shape[1] + 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(str(test_dir / f"{img_name}_test.jpg"), combined)
    
    print(f"\n✅ Calibration test saved to: {test_dir}/")
    print("   Check if the 'Calibrated' images look straight!")
    
    # 3. Process all images
    all_images = sorted(list(input_dir.glob("*.jpg")))[:20]  # First 20 for testing
    
    print(f"\n🔄 Processing {len(all_images)} images with calibration...")
    
    stats = {
        'total': len(all_images),
        'rotated': 0,
        'straight': 0,
        'angles': []
    }
    
    for i, img_path in enumerate(all_images):
        print(f"[{i+1}/{len(all_images)}] {img_path.name}")
        
        deskewed_img, final_angle, raw_angle = deskew_with_calibration(
            img_path, calibration_offset
        )
        
        if deskewed_img is not None:
            cv2.imwrite(str(output_dir / img_path.name), deskewed_img)
            
            if abs(final_angle) > 0.1:
                stats['rotated'] += 1
                stats['angles'].append(final_angle)
                print(f"  🔄 raw={raw_angle:.2f}° → final={final_angle:.2f}°")
            else:
                stats['straight'] += 1
                print(f"  ✓ raw={raw_angle:.2f}° (straight)")
        else:
            # Copy original
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(output_dir / img_path.name), img)
            print(f"  ⚠️  Could not process, copied original")
    
    # Print summary
    print(f"\n📊 Results with calibration offset {calibration_offset:.2f}°:")
    print(f"  Total: {stats['total']}")
    print(f"  Rotated: {stats['rotated']}")
    print(f"  Already straight: {stats['straight']}")
    
    if stats['angles']:
        print(f"  Average rotation: {np.mean(stats['angles']):.2f}°")
    
    print(f"\n✅ Calibrated images saved to: {output_dir}")
    
    return output_dir, calibration_offset

def manual_calibration_adjustment():
    """
    Manual mode: Let you adjust the calibration offset.
    """
    input_dir = Path("data/from_jeremy/images")
    
    # Test different offsets
    test_offsets = [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
    
    print("🎛️  MANUAL CALIBRATION ADJUSTMENT")
    print("="*60)
    
    # Pick a known problem image
    test_image = "m-t0627-00538-00654.jpg"  # Or any problem image
    
    img_path = input_dir / test_image
    if not img_path.exists():
        print(f"Test image not found: {test_image}")
        # Use first available
        images = list(input_dir.glob("*.jpg"))
        if images:
            img_path = images[0]
            test_image = img_path.name
    
    print(f"Testing on: {test_image}")
    
    original = cv2.imread(str(img_path))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    raw_angle = detect_skew_angle(gray)
    
    print(f"Raw detected angle: {raw_angle:.2f}°")
    print("\nTesting different calibration offsets:")
    
    test_dir = Path("manual_calibration_tests")
    test_dir.mkdir(exist_ok=True)
    
    for offset in test_offsets:
        calibrated_angle = raw_angle - offset
        final_angle = -calibrated_angle  # Opposite rotation
        
        if abs(final_angle) > 0.1:
            rotated = _rotate_image(original, final_angle)
            
            # Save with offset in filename
            filename = f"{img_path.stem}_offset{offset:+.1f}_final{final_angle:+.1f}.jpg"
            cv2.imwrite(str(test_dir / filename), rotated)
            
            print(f"  Offset {offset:+.1f}° → final={final_angle:+.1f}° → saved")
    
    print(f"\n✅ Test images saved to: {test_dir}/")
    print("   Check which offset produces the straightest image!")
    print("   Then use that offset in the main script.")

if __name__ == "__main__":
    print("="*70)
    print("🎯 CALIBRATION-BASED DESKEW - FINAL SOLUTION")
    print("="*70)
    
    print("\nThis script will:")
    print("1. Find images that are already straight (calibration set)")
    print("2. Calculate camera 'straight' offset")
    print("3. Apply calibration to all images")
    print("4. Save deskewed images")
    
    print("\nOptions:")
    print("1. Run full calibration + deskew (20 images test)")
    print("2. Manual calibration adjustment (find best offset)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        output_dir, offset = batch_calibrated_deskew()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Check 'calibration_test/' folder - do calibrated images look straight?")
        print(f"2. Check 'data/from_jeremy/images_calibrated_straight/'")
        print(f"3. If good, update script to process ALL images")
        print(f"4. Update extraction to use: {output_dir}/")
        
    elif choice == "2":
        manual_calibration_adjustment()
        
    else:
        print("Invalid choice")