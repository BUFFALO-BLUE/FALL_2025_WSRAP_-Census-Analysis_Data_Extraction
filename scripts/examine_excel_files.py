import pandas as pd
import os
import glob

def examine_excel_files():
    """Examine the structure of Jeremy's Excel files"""
    
    print("=== EXAMINING EXCEL FILES ===")
    
    # Check for Excel files
    excel_files = glob.glob('data/from_jeremy/transcriptions/*.xlsx') + \
                  glob.glob('data/from_jeremy/transcriptions/*.xls')
    
    if not excel_files:
        print("❌ No Excel files found in data/from_jeremy/transcriptions/")
        print("Please make sure Jeremy's Excel files are in that folder")
        return
    
    print(f"Found {len(excel_files)} Excel files:")
    for file in excel_files:
        print(f"  ✅ {os.path.basename(file)}")
    
    # Examine the first Excel file as an example
    first_excel = excel_files[0]
    print(f"\n--- Examining {os.path.basename(first_excel)} ---")
    
    try:
        # Try to read the Excel file
        df = pd.read_excel(first_excel)
        
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\nColumn names:")
        for col in df.columns:
            print(f"  - '{col}'")
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        # Check for specific columns we expect
        expected_columns = ['race', 'gender', 'marital status', 'wages', 'hours', 'house number']
        found_columns = []
        
        for expected in expected_columns:
            for actual in df.columns:
                if expected.lower() in str(actual).lower():
                    found_columns.append((expected, actual))
        
        if found_columns:
            print("\n✅ Found matching columns:")
            for expected, actual in found_columns:
                print(f"  {expected} -> '{actual}'")
        else:
            print("\n⚠️  No exact column matches found")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None

def compare_to_extracted_cells():
    """Compare Excel structure to our extracted cells"""
    
    print("\n=== COMPARING TO EXTRACTED CELLS ===")
    
    # Check how many images we extracted cells from
    extracted_folders = os.listdir('data/extracted_cells') if os.path.exists('data/extracted_cells') else []
    
    print(f"Extracted cells from {len(extracted_folders)} images")
    print("Each image should have matching data in Excel")
    
    # Check head rows count
    total_head_cells = 0
    for folder in extracted_folders[:3]:  # Check first 3
        head_dir = os.path.join('data/extracted_cells', folder, 'head_rows')
        if os.path.exists(head_dir):
            head_files = os.listdir(head_dir)
            print(f"  {folder}: {len(head_files)} head cells")
            total_head_cells += len(head_files)
    
    print(f"\nEstimated total head cells to label: ~{total_head_cells}")
    print("We need to map each of these to Excel labels")

if __name__ == "__main__":
    print("🔍 Analyzing Jeremy's Excel files...")
    
    df = examine_excel_files()
    compare_to_extracted_cells()
    
    if df is not None:
        print("\n✅ Excel file analysis complete!")
        print("\n📋 Next steps:")
        print("1. We'll create a mapping between Excel rows and our extracted rows")
        print("2. Copy labeled images to training folders")
        print("3. Create metadata CSV for training")
    else:
        print("\n❌ Could not analyze Excel files")