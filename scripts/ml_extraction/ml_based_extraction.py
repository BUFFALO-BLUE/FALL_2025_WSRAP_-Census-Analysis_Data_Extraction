"""
ML-BASED EXTRACTION USING SMART ADAPTIVE GRID DETECTION
1. Uses images_aligned_to_first (all images aligned and formatted the same)
2. Uses smart_adaptive_grid method (same as smart_adaptive_grid.png)
3. Use ML to detect '0' in head column
4. Extract data from rows with '0' (house number, price, rented/owned, etc.)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import sys
from datetime import datetime

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

# Column coordinates from smart_adaptive_extraction.py (matches smart_adaptive_grid.png)
SMART_ADAPTIVE_COLUMNS = {
    'street': (629, 718),
    'house_number': (718, 836),
    'rented_owned': (914, 994),
    'price_rent': (996, 1143),
    'head': (1889, 2204),  # Head column - where '0' detection happens
    'gender': (2204, 2285),
    'race': (2285, 2388),
    'marital_status': (2491, 2574),
    'hours_worked': (4939, 5092),
    'wages': (6433, 6588)
}


class MLZeroDetector:
    """Use TensorFlow/Keras to detect '0' in head column cells"""
    
    def __init__(self):
        self.model = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create simple CNN for '0' detection"""
        model_path = Path("models/zero_detector.h5")
        
        if model_path.exists():
            print("✅ Loading existing '0' detection model...")
            try:
                self.model = keras.models.load_model(str(model_path))
                print(f"   Model loaded from: {model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("   Creating new model...")
                self.model = self.create_simple_model()
        else:
            print("⚠️  No model found. Creating simple CNN...")
            print("   Train it first with: python scripts/ml_extraction/train_zero_detector.py")
            self.model = self.create_simple_model()
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(model_path))
            print(f"✅ Untrained model saved: {model_path}")
            print("   ⚠️  Model needs training before use!")
    
    def create_simple_model(self):
        """Create simple CNN for binary classification: '0' vs not '0'"""
        model = keras.Sequential([
            keras.layers.Input(shape=(50, 50, 1)),  # Head cell size
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')  # Binary: '0' or not
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_cell(self, cell_image):
        """Preprocess cell image for model input"""
        if cell_image.size == 0:
            return None
        
        # Resize to model input size
        resized = cv2.resize(cell_image, (50, 50))
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def detect_zero(self, cell_image):
        """Detect if cell contains '0'"""
        if cell_image.size == 0:
            return False, 0.0
        
        # Preprocess
        processed = self.preprocess_cell(cell_image)
        if processed is None:
            return False, 0.0
        
        # Predict
        try:
            prediction = self.model.predict(processed, verbose=0)[0][0]
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return False, 0.0
        
        # Threshold (adjust based on training)
        is_zero = prediction > 0.5
        confidence = prediction if is_zero else 1.0 - prediction
        
        return is_zero, float(confidence)


class ImprovedExtractor:
    """Improved extraction using smart_adaptive_grid method + ML"""
    
    def __init__(self, debug=False):
        self.zero_detector = MLZeroDetector()
        self.debug = debug
        # Smart adaptive parameters (same as smart_adaptive_grid.png)
        self.first_row_y = 1263
        self.expected_row_height = 78
        self.target_rows = 40
        self.head_x1, self.head_x2 = SMART_ADAPTIVE_COLUMNS['head']
    
    def extract_from_aligned_image(self, image_path, reference_image_path=None):
        """
        Extract data from aligned image using:
        1. Smart adaptive row detection (same as smart_adaptive_grid.png)
        2. ML-based '0' detection
        3. Extract fields from rows with '0'
        """
        print(f"\n📸 Processing: {Path(image_path).name}")
        
        # Step 1: Load image
        img = cv2.imread(str(image_path), 0)  # Grayscale
        if img is None:
            print("  ❌ Could not load image")
            return None
        
        # Step 2: Detect rows using smart adaptive method
        row_boundaries = find_smart_row_boundaries(
            img, self.first_row_y, self.expected_row_height, self.target_rows
        )
        
        print(f"  ✅ Detected {len(row_boundaries)-1} rows using smart_adaptive_grid method")
        
        # Step 3: Detect rows with '0' in head column
        head_rows = []
        
        print(f"  🔍 Scanning {len(row_boundaries)-1} rows for '0'...")
        
        for row_idx in range(len(row_boundaries) - 1):
            y1 = row_boundaries[row_idx]
            y2 = row_boundaries[row_idx + 1]
            
            # Skip very short rows
            if y2 - y1 < 20:
                continue
            
            # Extract head cell using smart_adaptive coordinates
            head_cell = img[y1:y2, self.head_x1:self.head_x2]
            
            if head_cell.size > 0:
                # Use ML to detect '0'
                is_zero, confidence = self.zero_detector.detect_zero(head_cell)
                
                if is_zero:
                    head_rows.append({
                        'row_idx': row_idx,
                        'y1': y1,
                        'y2': y2,
                        'confidence': confidence
                    })
                    if self.debug or row_idx % 5 == 0:  # Print every 5th or all if debug
                        print(f"    Row {row_idx}: '0' detected (confidence: {confidence:.2f})")
        
        print(f"  ✅ Found {len(head_rows)} head rows")
        
        # Step 4: Extract data from head rows using smart_adaptive column coordinates
        extracted_data = []
        
        for head_row in head_rows:
            row_data = self.extract_row_data(img, head_row)
            if row_data:
                extracted_data.append(row_data)
        
        return {
            'image': Path(image_path).name,
            'num_rows': len(row_boundaries) - 1,
            'head_rows': head_rows,
            'extracted_data': extracted_data
        }
    
    def extract_row_data(self, img, head_row):
        """Extract all fields from a head row using smart_adaptive column coordinates"""
        y1, y2 = head_row['y1'], head_row['y2']
        
        row_data = {
            'row_index': head_row['row_idx'],
            'confidence': head_row['confidence'],
            'cells': {}
        }
        
        # Extract all columns using smart_adaptive coordinates
        for col_name, (x1, x2) in SMART_ADAPTIVE_COLUMNS.items():
            cell_img = img[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                row_data['cells'][col_name] = cell_img
        
        return row_data


def batch_extract_with_ml(debug=False):
    """Process all aligned images with ML-based extraction"""
    
    print("="*80)
    print("🤖 ML-BASED EXTRACTION WITH SMART ADAPTIVE GRID")
    print("="*80)
    
    # Use aligned images
    input_dir = Path("data/from_jeremy/images_aligned_to_first")
    if not input_dir.exists():
        print(f"❌ Aligned images not found: {input_dir}")
        print("   Run first: python scripts/get_image_alignment_angle.py")
        return None
    
    image_paths = sorted(list(input_dir.glob("*.jpg")))
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        return None
    
    print(f"📸 Found {len(image_paths)} aligned images")
    
    # Initialize extractor
    extractor = ImprovedExtractor(debug=debug)
    
    # Create output directory
    output_dir = Path("data/ml_extracted_cells")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}]")
        
        result = extractor.extract_from_aligned_image(img_path)
        
        if result:
            # Save extracted cells
            image_output_dir = output_dir / Path(img_path).stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            cells_saved = 0
            for row_data in result['extracted_data']:
                row_idx = row_data['row_index']
                
                for field_name, cell_img in row_data['cells'].items():
                    filename = f"row{row_idx:02d}_{field_name}.png"
                    filepath = image_output_dir / filename
                    cv2.imwrite(str(filepath), cell_img)
                    cells_saved += 1
            
            result['cells_saved'] = cells_saved
            all_results.append(result)
            
            print(f"  ✅ Saved {cells_saved} cells")
        else:
            print(f"  ❌ Extraction failed")
    
    # Save summary
    summary = {
        'total_images': len(image_paths),
        'processed': len(all_results),
        'total_head_rows': sum(len(r['head_rows']) for r in all_results),
        'total_cells': sum(r.get('cells_saved', 0) for r in all_results),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"   Processed: {summary['processed']}/{summary['total_images']} images")
    print(f"   Total head rows: {summary['total_head_rows']}")
    print(f"   Total cells: {summary['total_cells']}")
    print(f"   Summary: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    print("="*80)
    print("🤖 ML-BASED CENSUS EXTRACTION")
    print("="*80)
    
    print("\nThis script:")
    print("1. Uses smart_adaptive_grid method (same as smart_adaptive_grid.png)")
    print("2. Uses ML to detect '0' in head column")
    print("3. Extracts data from rows with '0'")
    print("4. Works on images_aligned_to_first (all images aligned and formatted the same)")
    
    print("\n⚠️  NOTE: ML model needs training first!")
    print("   Use your calibration samples to train the '0' detector")
    print("   Run: python scripts/ml_extraction/train_zero_detector.py")
    
    print("\nOptions:")
    print("1. Run extraction (test mode - first 5 images)")
    print("2. Run extraction (full batch)")
    print("3. Run with debug output")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Test mode - just first 5 images
        input_dir = Path("data/from_jeremy/images_aligned_to_first")
        image_paths = sorted(list(input_dir.glob("*.jpg")))[:5]
        
        extractor = ImprovedExtractor(debug=True)
        output_dir = Path("data/ml_extracted_cells")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            result = extractor.extract_from_aligned_image(img_path)
            if result:
                image_output_dir = output_dir / Path(img_path).stem
                image_output_dir.mkdir(parents=True, exist_ok=True)
                
                cells_saved = 0
                for row_data in result['extracted_data']:
                    row_idx = row_data['row_index']
                    for field_name, cell_img in row_data['cells'].items():
                        filename = f"row{row_idx:02d}_{field_name}.png"
                        filepath = image_output_dir / filename
                        cv2.imwrite(str(filepath), cell_img)
                        cells_saved += 1
                
                print(f"  ✅ Saved {cells_saved} cells")
        
        print("\n✅ Test extraction complete!")
        
    elif choice == "2":
        results = batch_extract_with_ml(debug=False)
        if results:
            print(f"\n✅ Processed {len(results)} images")
    
    elif choice == "3":
        results = batch_extract_with_ml(debug=True)
        if results:
            print(f"\n✅ Processed {len(results)} images")
    
    else:
        print("Invalid choice. Running test mode...")
        # Run test mode
        input_dir = Path("data/from_jeremy/images_aligned_to_first")
        if input_dir.exists():
            image_paths = sorted(list(input_dir.glob("*.jpg")))[:1]
            extractor = ImprovedExtractor(debug=True)
            result = extractor.extract_from_aligned_image(image_paths[0])
            if result:
                print(f"\n✅ Test successful! Found {len(result['head_rows'])} head rows")



