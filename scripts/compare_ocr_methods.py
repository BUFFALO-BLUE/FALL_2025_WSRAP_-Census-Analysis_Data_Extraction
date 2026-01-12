"""
Compare Keras-OCR vs Tesseract OCR
Helps you decide which OCR method works better for your census data.
"""

import cv2
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json

# Try to import both OCR libraries
try:
    import keras_ocr
    KERAS_OCR_AVAILABLE = True
except ImportError:
    KERAS_OCR_AVAILABLE = False
    print("⚠️  keras-ocr not available. Install with: pip install keras-ocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not available. Install with: pip install pytesseract")

def preprocess_image(image_path: Path) -> tuple:
    """Preprocess image for OCR."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    return img, denoised

def ocr_with_keras(image_path: Path) -> dict:
    """OCR using keras-ocr."""
    if not KERAS_OCR_AVAILABLE:
        return {'text': '', 'time': 0, 'error': 'keras-ocr not available'}
    
    try:
        start_time = time.time()
        
        # Initialize pipeline (cached after first use)
        pipeline = keras_ocr.pipeline.Pipeline()
        
        # Read and preprocess
        img, processed = preprocess_image(image_path)
        if img is None:
            return {'text': '', 'time': 0, 'error': 'Could not read image'}
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        # Run OCR
        predictions = pipeline.recognize([img_rgb])[0]
        
        # Extract text
        text_parts = [text for text, _ in predictions]
        text = " ".join(text_parts)
        
        elapsed = time.time() - start_time
        
        return {
            'text': text.strip(),
            'time': elapsed,
            'num_detections': len(predictions),
            'error': None
        }
    except Exception as e:
        return {'text': '', 'time': 0, 'error': str(e)}

def ocr_with_tesseract(image_path: Path) -> dict:
    """OCR using Tesseract."""
    if not TESSERACT_AVAILABLE:
        return {'text': '', 'time': 0, 'error': 'pytesseract not available'}
    
    try:
        start_time = time.time()
        
        # Read and preprocess
        img, processed = preprocess_image(image_path)
        if img is None:
            return {'text': '', 'time': 0, 'error': 'Could not read image'}
        
        # Tesseract configuration for handwriting
        config = '--oem 3 --psm 7'  # Single text line
        
        # Run OCR
        text = pytesseract.image_to_string(processed, config=config).strip()
        
        # Get confidence
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        confidences = [c for c in data['conf'] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        elapsed = time.time() - start_time
        
        return {
            'text': text,
            'time': elapsed,
            'confidence': avg_confidence,
            'error': None
        }
    except Exception as e:
        return {'text': '', 'time': 0, 'error': str(e)}

def compare_methods(image_paths: list, limit: int = 20):
    """
    Compare Keras-OCR vs Tesseract on sample images.
    
    Parameters:
    -----------
    image_paths : list
        List of image paths to test
    limit : int
        Maximum number of images to test
    """
    print("="*80)
    print("🔍 COMPARING OCR METHODS: KERAS-OCR vs TESSERACT")
    print("="*80)
    
    if not KERAS_OCR_AVAILABLE and not TESSERACT_AVAILABLE:
        print("❌ Neither OCR library is available!")
        return
    
    # Limit images
    test_images = image_paths[:limit] if limit else image_paths
    print(f"\n📸 Testing on {len(test_images)} images...")
    
    results = []
    
    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] {img_path.name}")
        
        result = {
            'image': img_path.name,
            'image_path': str(img_path)
        }
        
        # Keras-OCR
        if KERAS_OCR_AVAILABLE:
            print("   Keras-OCR...", end=" ")
            keras_result = ocr_with_keras(img_path)
            result['keras_text'] = keras_result['text']
            result['keras_time'] = keras_result['time']
            result['keras_detections'] = keras_result.get('num_detections', 0)
            result['keras_error'] = keras_result.get('error')
            print(f"✓ ({keras_result['time']:.2f}s) -> '{keras_result['text'][:30]}...'")
        else:
            result['keras_text'] = ''
            result['keras_time'] = 0
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            print("   Tesseract...", end=" ")
            tess_result = ocr_with_tesseract(img_path)
            result['tesseract_text'] = tess_result['text']
            result['tesseract_time'] = tess_result['time']
            result['tesseract_confidence'] = tess_result.get('confidence', 0)
            result['tesseract_error'] = tess_result.get('error')
            print(f"✓ ({tess_result['time']:.2f}s, {tess_result.get('confidence', 0):.1f}%) -> '{tess_result['text'][:30]}...'")
        else:
            result['tesseract_text'] = ''
            result['tesseract_time'] = 0
        
        # Compare results
        if KERAS_OCR_AVAILABLE and TESSERACT_AVAILABLE:
            keras_text = keras_result['text'].lower().strip()
            tess_text = tess_result['text'].lower().strip()
            
            if keras_text == tess_text:
                result['match'] = 'exact'
            elif keras_text and tess_text and (keras_text in tess_text or tess_text in keras_text):
                result['match'] = 'partial'
            else:
                result['match'] = 'different'
            
            result['both_empty'] = (not keras_text and not tess_text)
            result['keras_only'] = (keras_text and not tess_text)
            result['tesseract_only'] = (tess_text and not keras_text)
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("data/ocr_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ocr_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    if KERAS_OCR_AVAILABLE:
        keras_avg_time = df['keras_time'].mean()
        keras_with_text = len(df[df['keras_text'].str.len() > 0])
        print(f"\n🔵 KERAS-OCR:")
        print(f"   Average time: {keras_avg_time:.2f} seconds per image")
        print(f"   Images with text detected: {keras_with_text}/{len(df)} ({100*keras_with_text/len(df):.1f}%)")
        print(f"   Average detections: {df['keras_detections'].mean():.1f}")
    
    if TESSERACT_AVAILABLE:
        tess_avg_time = df['tesseract_time'].mean()
        tess_with_text = len(df[df['tesseract_text'].str.len() > 0])
        tess_avg_conf = df['tesseract_confidence'].mean()
        print(f"\n🟢 TESSERACT:")
        print(f"   Average time: {tess_avg_time:.2f} seconds per image")
        print(f"   Images with text detected: {tess_with_text}/{len(df)} ({100*tess_with_text/len(df):.1f}%)")
        print(f"   Average confidence: {tess_avg_conf:.1f}%")
    
    if KERAS_OCR_AVAILABLE and TESSERACT_AVAILABLE:
        exact_matches = len(df[df['match'] == 'exact'])
        partial_matches = len(df[df['match'] == 'partial'])
        different = len(df[df['match'] == 'different'])
        
        print(f"\n🔄 AGREEMENT:")
        print(f"   Exact matches: {exact_matches}/{len(df)} ({100*exact_matches/len(df):.1f}%)")
        print(f"   Partial matches: {partial_matches}/{len(df)} ({100*partial_matches/len(df):.1f}%)")
        print(f"   Different results: {different}/{len(df)} ({100*different/len(df):.1f}%)")
        
        keras_only = len(df[df['keras_only'] == True])
        tess_only = len(df[df['tesseract_only'] == True])
        
        print(f"\n📝 TEXT DETECTION:")
        print(f"   Keras-OCR only: {keras_only} images")
        print(f"   Tesseract only: {tess_only} images")
        
        # Speed comparison
        if keras_avg_time > 0 and tess_avg_time > 0:
            speed_ratio = tess_avg_time / keras_avg_time
            if speed_ratio > 1:
                print(f"\n⚡ SPEED: Tesseract is {speed_ratio:.1f}x faster")
            else:
                print(f"\n⚡ SPEED: Keras-OCR is {1/speed_ratio:.1f}x faster")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if KERAS_OCR_AVAILABLE and TESSERACT_AVAILABLE:
        if keras_with_text > tess_with_text:
            print("   ✅ Keras-OCR detected text in more images")
        elif tess_with_text > keras_with_text:
            print("   ✅ Tesseract detected text in more images")
        else:
            print("   ✅ Both methods detected text in similar number of images")
        
        if exact_matches / len(df) > 0.7:
            print("   ✅ High agreement - both methods work well")
        elif different / len(df) > 0.5:
            print("   ⚠️  Low agreement - methods give different results")
            print("      Consider using both and comparing with ground truth")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Review the comparison CSV file")
    print("   2. Check images where methods disagree")
    print("   3. Compare with ground truth Excel data")
    print("   4. Choose the method that works best for your data")
    
    return df


if __name__ == "__main__":
    print("="*80)
    print("🔍 OCR METHOD COMPARISON TOOL")
    print("="*80)
    
    # Find cell images
    cells_dir = Path("data/extracted_cells")
    
    if not cells_dir.exists():
        print(f"❌ Directory not found: {cells_dir}")
        print("   Please run extraction first!")
        exit(1)
    
    # Get image paths
    image_paths = sorted(list(cells_dir.glob("*.png")))
    
    if not image_paths:
        print(f"❌ No PNG images found in {cells_dir}")
        exit(1)
    
    print(f"\n📁 Found {len(image_paths)} images in {cells_dir}")
    
    # Ask user
    print("\nChoose test size:")
    print("1. Quick test (10 images)")
    print("2. Medium test (50 images)")
    print("3. Full test (all images)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        limit = 10
    elif choice == "2":
        limit = 50
    elif choice == "3":
        limit = None
    else:
        print("Invalid choice. Using quick test (10 images)...")
        limit = 10
    
    # Run comparison
    results_df = compare_methods(image_paths, limit=limit)
    
    print("\n✅ Comparison complete!")








