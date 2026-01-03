# scripts/fixed_direction_deskew.py
"""
DESKEW WITH CORRECT DIRECTION: Flips the rotation sign to make images straight.
"""

import cv2
import numpy as np
from pathlib import Path
import math

def deskew_correct_direction(image_path, output_path=None):
    """
    Deskew with CORRECTED direction - flips the sign of rotation.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # METHOD: Find lines and determine correct rotation direction
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) < 5:
        print(f"  ⚠️  Not enough lines, trying text detection")
        return _deskew_by_text_with_sign_check(gray, original)
    
    # Analyze line angles
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
        median_angle = np.median(angles)
        
        # 🔥 CRITICAL FIX: Flip the sign of the rotation!
        # If image was rotated wrong way before, flip it now
        corrected_angle = -median_angle  # Just flip the sign!
        
        if abs(corrected_angle) > 0.5:
            print(f"  📐 Original detection: {median_angle:.2f}°")
            print(f"  🔄 Corrected rotation: {corrected_angle:.2f}° (sign flipped)")
            
            deskewed_img = _rotate_image(original, corrected_angle)
            return deskewed_img, corrected_angle
    
    print(f"  ✓ Image appears straight")
    return original, 0.0

def _deskew_by_text_with_sign_check(gray_img, original_img):
    """Text-based deskewing with sign check"""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
            
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        
        if angle < -45:
            angle = 90 + angle
        
        if -45 <= angle <= 45:
            angles.append(angle)
    
    if angles:
        median_angle = np.median(angles)
        corrected_angle = -median_angle  # Flip sign
        
        if abs(corrected_angle) > 0.5:
            print(f"  📝 Text detection: {median_angle:.2f}° → {corrected_angle:.2f}° (sign flipped)")
            return _rotate_image(original_img, corrected_angle), corrected_angle
    
    return original_img, 0.0

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

def test_with_manual_sign(input_dir, output_dir, test_count=5):
    """
    Test with manual sign adjustment - try both positive and negative rotations.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(input_path.glob("*.jpg")))[:test_count]
    
    print("🧪 TESTING ROTATION DIRECTIONS")
    print("="*60)
    print("We'll try both +angle and -angle to see which looks straight")
    print("="*60)
    
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get detected angle (without correction)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            print(f"  No lines detected, copying original")
            cv2.imwrite(str(output_path / img_path.name), img)
            continue
        
        # Calculate raw angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                angle = 90.0
            else:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if angles:
            raw_angle = np.median(angles)
            
            # Try BOTH directions
            test_angles = [raw_angle, -raw_angle]
            
            print(f"  Raw detected angle: {raw_angle:.2f}°")
            print(f"  Testing rotations: {test_angles[0]:.2f}° and {test_angles[1]:.2f}°")
            
            # Save all test versions
            for j, test_angle in enumerate(test_angles):
                if abs(test_angle) > 0.5:
                    rotated = _rotate_image(img, test_angle)
                    
                    # Save with angle in filename
                    test_name = f"{img_path.stem}_test{j+1}_{test_angle:+.1f}deg.jpg"
                    cv2.imwrite(str(output_path / test_name), rotated)
                    
                    print(f"    Saved: {test_name}")
            
            # Also save original for comparison
            orig_name = f"{img_path.stem}_original.jpg"
            cv2.imwrite(str(output_path / orig_name), img)
    
    print(f"\n✅ Created test images in: {output_dir}")
    print("   Check which version (test1 or test2) looks straight!")
    print("   The correct one should have vertical/horizontal lines aligned.")

def process_with_sign_flip(input_dir, output_dir, max_images=None):
    """
    Process all images with sign flip (using -angle instead of angle).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(input_path.glob("*.jpg")))
    
    if max_images:
        images = images[:max_images]
    
    print("🔄 PROCESSING WITH SIGN FLIP (-angle)")
    print("="*60)
    
    rotated_count = 0
    
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name}")
        
        deskewed_img, angle = deskew_correct_direction(img_path)
        
        if deskewed_img is not None:
            cv2.imwrite(str(output_path / img_path.name), deskewed_img)
            
            if abs(angle) > 0.1:
                rotated_count += 1
                print(f"  ✅ Rotated {angle:.1f}°")
            else:
                print(f"  ✓ Straight")
        else:
            # Copy original
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(output_path / img_path.name), img)
            print(f"  ⚠️  Could not process, copied original")
    
    print(f"\n✅ Processed {len(images)} images")
    print(f"   Rotated: {rotated_count}")
    print(f"   Output: {output_dir}")
    
    return output_dir

def simple_visual_check():
    """Quick visual check - creates comparison images"""
    input_dir = Path("data/from_jeremy/images")
    output_dir = Path("sign_flip_test")
    output_dir.mkdir(exist_ok=True)
    
    images = sorted(list(input_dir.glob("*.jpg")))[:3]
    
    print("👀 CREATING VISUAL COMPARISON")
    print("="*60)
    
    for img_path in images:
        original = cv2.imread(str(img_path))
        
        # Get raw angle first
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        raw_angle = 0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                raw_angle = np.median(angles)
        
        # Create 3 versions: original, +angle, -angle
        versions = [
            ("Original", original, 0),
            (f"+{abs(raw_angle):.1f}°", _rotate_image(original, abs(raw_angle)), abs(raw_angle)),
            (f"-{abs(raw_angle):.1f}°", _rotate_image(original, -abs(raw_angle)), -abs(raw_angle))
        ]
        
        # Resize for comparison
        resized_versions = []
        for name, img, angle in versions:
            scale = 600 / img.shape[0]
            resized = cv2.resize(img, (int(img.shape[1] * scale), 600))
            resized_versions.append((name, resized, angle))
        
        # Combine horizontally
        combined = np.hstack([img for _, img, _ in resized_versions])
        
        # Add labels
        for j, (name, _, angle) in enumerate(resized_versions):
            x_pos = j * resized_versions[0][1].shape[1] + 50
            color = (0, 0, 255) if j == 0 else ((0, 255, 0) if j == 2 else (255, 0, 0))
            cv2.putText(combined, name, (x_pos, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Save
        output_path = output_dir / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(output_path), combined)
        
        print(f"  Saved: {output_path.name}")
        print(f"    Raw angle: {raw_angle:.2f}°")
    
    print(f"\n🎯 Check 'sign_flip_test/' folder")
    print("   Left: Original, Middle: +angle, Right: -angle")
    print("   Which middle or right looks straight?")
    print("   If RIGHT looks straight, we need negative rotation!")
    
    return output_dir

if __name__ == "__main__":
    print("🔥 DESKEW WITH CORRECTED ROTATION DIRECTION")
    print("="*70)
    
    print("\nOptions:")
    print("1. Quick visual test (see +angle vs -angle)")
    print("2. Test with manual adjustments (5 images)")
    print("3. Process all with sign flip")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        output_dir = simple_visual_check()
        
        print(f"\n🔍 After checking, which looks straight?")
        print("   If the RIGHT image (-angle) looks straight:")
        print("   Use option 3 to process all images with sign flip!")
        
    elif choice == "2":
        test_with_manual_sign(
            input_dir="data/from_jeremy/images",
            output_dir="data/deskew_direction_test",
            test_count=5
        )
        
        print(f"\n🎯 Check 'data/deskew_direction_test/' folder")
        print("   Look for files ending with '_test1_...' and '_test2_...'")
        print("   Determine which test looks straight!")
        
    elif choice == "3":
        print("\n🔄 Processing all images with sign flip")
        print("   This uses -angle instead of angle")
        print("   (Assuming -angle is correct based on visual tests)")
        
        output_dir = process_with_sign_flip(
            input_dir="data/from_jeremy/images",
            output_dir="data/from_jeremy/images_corrected_straight",
            max_images=20  # Start with 20 for testing
        )
        
        print(f"\n✅ Check results in: {output_dir}")
        print("   If they look good, remove max_images limit in script")
        print("   Then update extraction to use: {output_dir}/")
        
    else:
        print("Invalid choice")