import pandas as pd
import os
import glob
import sys

def install_openpyxl():
    """Install openpyxl if not available"""
    try:
        import openpyxl
    except ImportError:
        print("Installing openpyxl...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        import openpyxl

def examine_excel_detailed():
    """Detailed examination of Excel file structure"""
    
    print("=== DETAILED EXCEL FILE ANALYSIS ===")
    
    # Check for Excel files
    excel_files = glob.glob('data/from_jeremy/transcriptions/*.xlsx')
    
    if not excel_files:
        print("❌ No Excel files found")
        return None
    
    excel_path = excel_files[0]
    print(f"Found Excel file: {os.path.basename(excel_path)}")
    print(f"File size: {os.path.getsize(excel_path) / (1024*1024):.2f} MB")
    
    try:
        # Read Excel file
        print("\nReading Excel file...")
        xl = pd.ExcelFile(excel_path)
        
        print(f"Number of sheets: {len(xl.sheet_names)}")
        print("Sheet names:", xl.sheet_names)
        
        # Read first sheet
        df = xl.parse(xl.sheet_names[0])
        
        print(f"\nSheet '{xl.sheet_names[0]}' has {df.shape[0]} rows and {df.shape[1]} columns")
        
        print("\n=== COLUMN NAMES ===")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. '{col}'")
        
        print("\n=== SAMPLE DATA (first 5 rows) ===")
        print(df.head())
        
        print("\n=== DATA TYPES ===")
        print(df.dtypes)
        
        print("\n=== COLUMN STATISTICS ===")
        for col in df.columns[:10]:  # Show first 10 columns
            print(f"\nColumn: '{col}'")
            print(f"  Non-null values: {df[col].count()} / {len(df)}")
            print(f"  Unique values: {df[col].nunique()}")
            if df[col].nunique() < 20:  # If few unique values, show them
                print(f"  Values: {df[col].unique()}")
        
        # Look for potential image name column
        print("\n=== LOOKING FOR IMAGE NAME COLUMN ===")
        potential_image_cols = []
        for col in df.columns:
            # Check if column contains image-like names
            sample = df[col].dropna().head(100)
            if any('m-t' in str(val) or '.jpg' in str(val) or '00634' in str(val) for val in sample):
                potential_image_cols.append(col)
                print(f"✅ Potential image column: '{col}'")
        
        # Look for our expected columns
        print("\n=== LOOKING FOR EXPECTED COLUMNS ===")
        expected_patterns = {
            'house': ['house', 'dwelling', 'number'],
            'race': ['race', 'color', 'ethnic'],
            'gender': ['gender', 'sex'],
            'marital': ['marital', 'married', 'single'],
            'hours': ['hour', 'work'],
            'wages': ['wage', 'salary', 'income'],
            'rent': ['rent', 'price', 'cost'],
            'head': ['head', 'relationship']
        }
        
        for field, patterns in expected_patterns.items():
            matching_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in patterns):
                    matching_cols.append(col)
            
            if matching_cols:
                print(f"✅ {field}: {matching_cols}")
            else:
                print(f"❌ {field}: No matching column found")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return None

def compare_to_extracted():
    """Compare Excel data to extracted cells"""
    
    print("\n=== COMPARING TO EXTRACTED CELLS ===")
    
    # Count extracted images
    if os.path.exists('data/extracted_cells'):
        extracted_folders = os.listdir('data/extracted_cells')
        print(f"Extracted cells from {len(extracted_folders)} images")
        
        # Sample a few folders to check head rows
        sample_count = min(5, len(extracted_folders))
        head_counts = []
        
        for folder in extracted_folders[:sample_count]:
            head_dir = os.path.join('data/extracted_cells', folder, 'head_rows')
            if os.path.exists(head_dir):
                head_files = os.listdir(head_dir)
                head_counts.append(len(head_files) / 9)  # Divide by number of columns
        
        if head_counts:
            avg_heads = sum(head_counts) / len(head_counts)
            print(f"Average head rows per image: {avg_heads:.1f}")
            print(f"Estimated total head rows: {avg_heads * len(extracted_folders):.0f}")
    else:
        print("❌ data/extracted_cells/ folder not found")

if __name__ == "__main__":
    # Install openpyxl if needed
    try:
        import openpyxl
    except ImportError:
        install_openpyxl()
    
    # Run analysis
    df = examine_excel_detailed()
    compare_to_extracted()
    
    if df is not None:
        print("\n✅ Analysis complete!")
        print("\n📋 Next steps:")
        print("1. Identify which Excel columns match our extracted columns")
        print("2. Find the column that links Excel rows to specific census images")
        print("3. Create mapping between Excel data and extracted cells")