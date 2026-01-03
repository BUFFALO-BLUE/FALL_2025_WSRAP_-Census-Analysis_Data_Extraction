import pytesseract
import pandas as pd
import cv2
import os
from pathlib import Path
import numpy as np

# Configure Tesseract path (UPDATE THIS if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_single_image_handwriting():
    """Read cursive handwriting from extracted cells and create Excel"""
    
    print("=" * 70)
    print("📝 OCR CURSIVE HANDWRITING FROM CENSUS CELLS")
    print("=" * 70)
    
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is available")
    except:
        print("❌ Tesseract not found. Please install:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install and add to PATH")
        print("   3. Or set pytesseract.pytesseract.tesseract_cmd = r'C:\\Path\\To\\tesseract.exe'")
        return
    
    # 1. Find extracted cells (check both possible locations)
    cells_dirs = [
        Path("data/extracted_cells/head_rows"),
        Path("data/processed/sample_extraction")  # If from smart_adaptive_extraction.py
    ]
    
    cells_dir = None
    for d in cells_dirs:
        if d.exists() and any(d.glob("*.png")):
            cells_dir = d
            break
    
    if cells_dir is None:
        print("❌ No extracted cell images found!")
        print("   Run: python scripts/smart_adaptive_extraction.py first")
        return
    
    print(f"📁 Found cells in: {cells_dir}")
    
    # 2. List all PNG files
    png_files = list(cells_dir.glob("*.png"))
    print(f"📸 Found {len(png_files)} cell images")
    
    if len(png_files) == 0:
        print("❌ No PNG files found")
        return
    
    # 3. Create OCR results list
    ocr_results = []
    
    print("\n🔍 Processing OCR on cell images...")
    
    for i, png_path in enumerate(png_files[:50]):  # Process first 50 cells
        # Parse filename to get row and field info
        filename = png_path.stem  # Remove .png extension
        
        # Different filename patterns
        if filename.startswith("HEAD_row"):
            # Pattern: HEAD_row00_race.png
            parts = filename.split("_")
            if len(parts) >= 3:
                row_num = parts[1].replace("row", "")
                field = parts[2]
            else:
                row_num = "unknown"
                field = "unknown"
        elif filename.startswith("cell_"):
            # Pattern: cell_0000_r4c19.png
            # Extract row and column from filename
            row_num = "unknown"
            field = "unknown"
            if "_r" in filename and "_c" in filename:
                try:
                    # Extract numbers after _r and _c
                    r_idx = filename.find("_r") + 2
                    c_idx = filename.find("_c")
                    row_num = filename[r_idx:c_idx]
                except:
                    pass
        else:
            row_num = "unknown"
            field = filename
        
        # Read image
        try:
            # Load image with OpenCV
            image = cv2.imread(str(png_path))
            
            if image is None:
                print(f"   ❌ Could not read: {png_path.name}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing for better OCR
            # 1. Thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # 2. Denoising
            denoised = cv2.medianBlur(thresh, 3)
            
            # 3. Try different OCR configurations for cursive handwriting
            # Configuration 1: Default
            custom_config = '--oem 3 --psm 7'  # Page segmentation mode 7: single text line
            
            try:
                # Try with default
                text_default = pytesseract.image_to_string(denoised, config=custom_config).strip()
                
                # Try with different PSM modes
                text_alt = pytesseract.image_to_string(denoised, config='--oem 3 --psm 8').strip()
                
                # Use the longer text (more likely to have content)
                if len(text_alt) > len(text_default):
                    text = text_alt
                else:
                    text = text_default
                    
                # Clean the text
                text = text.replace('\n', ' ').replace('\r', '').strip()
                
                # Get confidence
                data = pytesseract.image_to_data(denoised, config=custom_config, output_type=pytesseract.Output.DICT)
                if data['conf']:
                    avg_conf = sum([c for c in data['conf'] if c > 0]) / len([c for c in data['conf'] if c > 0]) if [c for c in data['conf'] if c > 0] else 0
                else:
                    avg_conf = 0
                
                # If text is empty but confidence is high, it might be numbers
                if not text and avg_conf > 50:
                    # Try digits only
                    text_digits = pytesseract.image_to_string(denoised, config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789').strip()
                    if text_digits:
                        text = text_digits
                
                ocr_results.append({
                    'Cell_Image': png_path.name,
                    'Row': row_num,
                    'Field': field,
                    'OCR_Text': text,
                    'Confidence': round(avg_conf, 1),
                    'Image_Path': str(png_path)
                })
                
                if i % 10 == 0:
                    print(f"   Processed {i+1}/{min(50, len(png_files))}: {png_path.name} -> '{text[:20]}...'")
                    
            except Exception as e:
                print(f"   ❌ OCR failed for {png_path.name}: {e}")
                ocr_results.append({
                    'Cell_Image': png_path.name,
                    'Row': row_num,
                    'Field': field,
                    'OCR_Text': 'OCR_ERROR',
                    'Confidence': 0,
                    'Image_Path': str(png_path)
                })
                
        except Exception as e:
            print(f"   ❌ Image processing failed for {png_path.name}: {e}")
    
    print(f"\n✅ OCR completed for {len(ocr_results)} cells")
    
    if len(ocr_results) == 0:
        print("❌ No OCR results generated")
        return
    
    # 4. Create DataFrame and organize by row and field
    df = pd.DataFrame(ocr_results)
    
    # Sort by row number if possible
    try:
        # Extract numeric row for sorting
        df['Row_Num'] = df['Row'].apply(lambda x: int(x) if x.isdigit() else 999)
        df = df.sort_values(['Row_Num', 'Field'])
        df = df.drop('Row_Num', axis=1)
    except:
        # If row parsing fails, just sort by filename
        df = df.sort_values('Cell_Image')
    
    # 5. Create Excel file with multiple sheets
    excel_path = r"data\extracted_cells\census_handwriting_ocr.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: All OCR Results
        df.to_excel(writer, sheet_name='All_OCR_Results', index=False)
        
        # Sheet 2: Grouped by Row (if we have row information)
        if 'Row' in df.columns and any(df['Row'] != 'unknown'):
            # Pivot to have fields as columns
            try:
                pivot_df = df.pivot_table(
                    index='Row',
                    columns='Field',
                    values='OCR_Text',
                    aggfunc='first'
                ).reset_index()
                pivot_df.to_excel(writer, sheet_name='By_Row', index=False)
            except:
                # If pivot fails, just write sorted by row
                df_sorted = df.sort_values('Row')
                df_sorted.to_excel(writer, sheet_name='By_Row', index=False)
        
        # Sheet 3: High Confidence Results (>70%)
        high_conf = df[df['Confidence'] > 70]
        high_conf.to_excel(writer, sheet_name='High_Confidence', index=False)
        
        # Sheet 4: Statistics
        stats_data = {
            'Metric': [
                'Total Cells Processed',
                'Cells with Text Found',
                'Average Confidence',
                'High Confidence Cells (>70%)',
                'Low Confidence Cells (<30%)',
                'Common Fields Found'
            ],
            'Value': [
                len(df),
                len(df[df['OCR_Text'] != '']),
                f"{df['Confidence'].mean():.1f}%",
                len(high_conf),
                len(df[df['Confidence'] < 30]),
                ', '.join(df['Field'].unique()[:5])
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print(f"\n✅ Excel file created: {excel_path}")
    print(f"   Contains {len(df)} OCR results")
    print(f"   Average confidence: {df['Confidence'].mean():.1f}%")
    
    # 6. Create a simplified CSV too
    csv_path = r"data\extracted_cells\census_ocr_simple.csv"
    simple_cols = ['Row', 'Field', 'OCR_Text', 'Confidence']
    if all(col in df.columns for col in simple_cols):
        df[simple_cols].to_csv(csv_path, index=False)
        print(f"✅ CSV file created: {csv_path}")
    
    # 7. Show sample results
    print("\n🔍 SAMPLE OCR RESULTS:")
    print("-" * 50)
    
    sample = df.head(15)
    for _, row in sample.iterrows():
        if row['OCR_Text']:
            print(f"Row {row['Row']} | {row['Field']}: '{row['OCR_Text']}' (Confidence: {row['Confidence']}%)")
    
    # 8. Create comparison with Jeremy's data if available
    jeremy_csv = r"data\from_jeremy\transcriptions\clean_census_data.csv"
    if os.path.exists(jeremy_csv):
        print("\n" + "=" * 70)
        print("📊 COMPARING OCR WITH JEREMY'S TRANSCRIPTIONS")
        print("=" * 70)
        
        jeremy_df = pd.read_csv(jeremy_csv)
        
        # Create comparison sheet
        comparison_data = []
        
        # Take first few rows for comparison
        for i in range(min(10, len(df), len(jeremy_df))):
            ocr_row = df.iloc[i] if i < len(df) else {}
            jeremy_row = jeremy_df.iloc[i] if i < len(jeremy_df) else {}
            
            comparison_data.append({
                'Row_Number': i+1,
                'OCR_Race': ocr_row.get('OCR_Text', '') if ocr_row.get('Field') == 'race' else '',
                'Jeremy_Race': jeremy_row.get('Race', '') if not pd.isna(jeremy_row.get('Race')) else '',
                'OCR_House': ocr_row.get('OCR_Text', '') if ocr_row.get('Field') == 'house_number' else '',
                'Jeremy_House': str(jeremy_row.get('House_Number', '')) if not pd.isna(jeremy_row.get('House_Number')) else '',
                'OCR_Street': ocr_row.get('OCR_Text', '') if ocr_row.get('Field') == 'street' else '',
                'Jeremy_Street': jeremy_row.get('Street_Name', '') if not pd.isna(jeremy_row.get('Street_Name')) else '',
                'OCR_Confidence': ocr_row.get('Confidence', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add to Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            comparison_df.to_excel(writer, sheet_name='OCR_vs_Jeremy', index=False)
        
        print(f"✅ Added comparison sheet to Excel")
        print("\nComparison of first 5 rows:")
        print(comparison_df.head().to_string())
    
    return df, excel_path

def preprocess_for_better_ocr():
    """Show how to preprocess images for better cursive recognition"""
    
    print("\n" + "=" * 70)
    print("🛠️  IMPROVING OCR FOR CURSIVE HANDWRITING")
    print("=" * 70)
    
    tips = """
TIPS FOR BETTER CURSIVE HANDWRITING OCR:

1. IMAGE PREPROCESSING:
   - Increase contrast: Make black darker, white brighter
   - Remove noise: Use median filter or Gaussian blur
   - Deskew: Straighten tilted text
   - Scale up: Make text larger (2x or 3x)

2. TESSERACT CONFIGURATION:
   - Use --psm 7 for single text line
   - Try --psm 8 for single word
   - Use --oem 3 for LSTM engine (best for handwriting)
   - Add custom word lists if you know expected values

3. FOR CENSUS-SPECIFIC TEXT:
   - Train Tesseract on similar cursive handwriting
   - Create custom dictionary of census terms
   - Use language model: 'eng' + 'script/Latin'

4. POST-PROCESSING:
   - Correct common OCR errors
   - Use context (e.g., house numbers are numeric)
   - Validate against known patterns

QUICK PREPROCESSING SCRIPT EXAMPLE:
```python
import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # Threshold
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Scale up (200%)
    height, width = cleaned.shape
    scaled = cv2.resize(cleaned, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    return scaled