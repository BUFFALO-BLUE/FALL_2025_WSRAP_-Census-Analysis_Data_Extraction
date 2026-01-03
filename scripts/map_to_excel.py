# scripts/map_to_excel_fixed.py
import pandas as pd
import shutil
import json
from pathlib import Path
import glob

def map_extracted_to_excel(extraction_suffix="aligned"):
    """
    Map extracted PNGs to Excel using Jeremy's "back to front" order.
    
    Parameters:
    -----------
    extraction_suffix : str
        Suffix for extracted folder (e.g., "aligned", "original", "ready")
    """
    
    print("="*70)
    print(f"📊 CENSUS DATA MAPPING: PNGs → EXCEL (using {extraction_suffix} images)")
    print("="*70)
    
    # Load cleaned Excel data
    excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
    if not excel_path.exists():
        print(f"❌ Excel file not found: {excel_path}")
        return []
    
    df_excel = pd.read_csv(excel_path)
    
    print(f"Excel has {len(df_excel)} rows")
    print(f"Excel columns: {list(df_excel.columns)}")
    
    # Get all extracted images (using specified extraction folder)
    extracted_dir = Path(f"data/extracted_cells_{extraction_suffix}")
    
    if not extracted_dir.exists():
        print(f"❌ Extracted directory not found: {extracted_dir}")
        print(f"Available extraction folders:")
        for folder in Path("data").glob("extracted_cells_*"):
            if folder.is_dir():
                print(f"  - {folder}")
        return []
    
    # Get all image folders (each census page)
    image_folders = sorted([f for f in extracted_dir.iterdir() if f.is_dir()])
    
    if not image_folders:
        print(f"❌ No image folders found in {extracted_dir}")
        return []
    
    print(f"Found {len(image_folders)} extracted image folders in {extracted_dir.name}")
    
    # Jeremy said "back to front" - Excel starts with LAST image, LAST row
    image_folders_reversed = list(reversed(image_folders))
    
    # Create training dataset with suffix
    train_dir = Path(f"data/training_dataset_{extraction_suffix}")
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze Excel columns to suggest mappings
    print(f"\n🔍 ANALYZING EXCEL COLUMNS...")
    excel_columns = list(df_excel.columns)
    
    # Common column mappings (adjust based on your Excel)
    # This should be customized based on your actual Excel structure
    column_mapping = {
        'race': 'Race',
        'house_number': 'House_Number', 
        'rented_owned': 'Rented',  # Or 'Owned_Home_Value' depending on Excel
        'price_rent': 'Owned_Home_Value',  # Or 'Rented' depending on Excel
        'head': 'Notes',  # Head names might be in Notes
        'gender': 'Gender',
        'marital_status': 'Marital_Status',
        'hours_worked': 'Hours_Worked',
        'wages': 'Wages'
    }
    
    # Auto-detect which columns actually exist in Excel
    actual_mappings = {}
    for png_cat, possible_excel_col in column_mapping.items():
        if possible_excel_col in excel_columns:
            actual_mappings[png_cat] = possible_excel_col
            print(f"  ✓ {png_cat} → {possible_excel_col}")
        else:
            # Try to find similar columns
            for col in excel_columns:
                if png_cat.lower() in col.lower() or col.lower() in png_cat.lower():
                    actual_mappings[png_cat] = col
                    print(f"  ~ {png_cat} → {col} (auto-matched)")
                    break
    
    # Let user confirm or modify mappings
    print(f"\n🤔 Do you want to use these mappings?")
    print(f"If not, please modify the 'column_mapping' dictionary in the script.")
    confirm = input("Continue with these mappings? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("Please update the column_mapping dictionary and run again.")
        return []
    
    # Create folders for each category
    for category in actual_mappings.keys():
        (train_dir / category).mkdir(exist_ok=True)
    
    # Also create a combined folder for all labeled data
    all_labeled_dir = train_dir / "all_labeled_cells"
    all_labeled_dir.mkdir(exist_ok=True)
    
    all_mappings = []
    excel_row_idx = 0
    
    print(f"\n📋 Mapping Excel rows to extracted cells...")
    print(f"  (Using 'back to front' order: Excel starts with last image, last row)")
    
    # Go through images in reverse order (back to front)
    for img_idx, img_folder in enumerate(image_folders_reversed):
        image_name = img_folder.name  # e.g., "m-t0627-00538-00741"
        
        # Get head rows for this image
        head_rows_dir = img_folder / "head_rows"
        if not head_rows_dir.exists():
            print(f"  ⚠️  No head_rows folder in {image_name}, skipping")
            continue
        
        # Count head rows in this image
        race_files = list(head_rows_dir.glob("HEAD_row*_race.png"))
        if not race_files:
            print(f"  ⚠️  No race files in {image_name}/head_rows, skipping")
            continue
        
        # Get unique row numbers from this image
        row_numbers = sorted(list(set([
            int(f.stem.split('_')[1].replace('row', '')) 
            for f in race_files
        ])))
        
        print(f"  📄 {image_name}: {len(row_numbers)} head rows found")
        
        # Go through rows in reverse order (back to front)
        for row_num in reversed(row_numbers):
            if excel_row_idx >= len(df_excel):
                print(f"  ⚠️  Reached end of Excel data (row {excel_row_idx})")
                break
            
            # Get Excel data for this row
            excel_row = df_excel.iloc[excel_row_idx]
            
            # Map each category
            row_mapped = False
            for png_category, excel_column in actual_mappings.items():
                # Find the PNG file
                png_file = head_rows_dir / f"HEAD_row{row_num:02d}_{png_category}.png"
                
                if png_file.exists():
                    # Get the Excel value for this column
                    excel_value = excel_row.get(excel_column, None)
                    
                    if pd.notna(excel_value) and str(excel_value).strip():
                        # Create new filename with label
                        label = str(excel_value)
                        
                        # Clean label for filename
                        clean_label = "".join(
                            c for c in label 
                            if c.isalnum() or c in (' ', '-', '_', '.', ',')
                        ).strip().replace(' ', '_')
                        
                        # Limit length and remove consecutive underscores
                        clean_label = clean_label[:50]
                        while '__' in clean_label:
                            clean_label = clean_label.replace('__', '_')
                        
                        # Create unique filename
                        new_filename = f"{image_name}_row{row_num:02d}_{png_category}_{clean_label}.png"
                        dest_path = train_dir / png_category / new_filename
                        
                        # Also copy to combined folder
                        combined_dest = all_labeled_dir / new_filename
                        
                        try:
                            # Copy the PNG
                            shutil.copy2(png_file, dest_path)
                            shutil.copy2(png_file, combined_dest)
                            
                            all_mappings.append({
                                'excel_row': excel_row_idx,
                                'image': image_name,
                                'row': row_num,
                                'png_category': png_category,
                                'excel_column': excel_column,
                                'label': label,
                                'clean_label': clean_label,
                                'png_source': str(png_file),
                                'png_dest': str(dest_path)
                            })
                            
                            row_mapped = True
                            
                        except Exception as e:
                            print(f"    ❌ Error copying {png_category}: {e}")
            
            if row_mapped:
                excel_row_idx += 1
            
            # Show progress every 10 rows
            if excel_row_idx % 10 == 0 and excel_row_idx > 0:
                print(f"    Progress: {excel_row_idx}/{len(df_excel)} Excel rows mapped")
        
        if excel_row_idx >= len(df_excel):
            print(f"  ✅ Finished mapping all Excel rows")
            break
    
    # Save mappings to CSV
    if all_mappings:
        df_mappings = pd.DataFrame(all_mappings)
        mappings_csv = train_dir / "training_mappings.csv"
        df_mappings.to_csv(mappings_csv, index=False)
        
        # Also save as JSON for easier loading
        mappings_json = train_dir / "training_mappings.json"
        with open(mappings_json, 'w') as f:
            json.dump(all_mappings, f, indent=2)
    
    print(f"\n✅ MAPPING COMPLETE!")
    print(f"Total mappings created: {len(all_mappings)}")
    print(f"Training dataset saved to: {train_dir}")
    
    # Show summary by category
    print(f"\n📊 SUMMARY BY CATEGORY:")
    category_counts = {}
    for mapping in all_mappings:
        cat = mapping['png_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} samples")
    
    # Show unique label counts
    print(f"\n🎯 UNIQUE LABELS PER CATEGORY:")
    for cat in sorted(category_counts.keys()):
        labels = set([m['label'] for m in all_mappings if m['png_category'] == cat])
        print(f"  {cat}: {len(labels)} unique labels")
        # Show first few labels
        sample_labels = list(labels)[:5]
        if sample_labels:
            print(f"    Sample: {', '.join(sample_labels[:3])}...")
    
    # Show sample mappings
    print(f"\n🎯 SAMPLE MAPPINGS (first 3):")
    for i, mapping in enumerate(all_mappings[:3]):
        print(f"{i+1}. Excel row {mapping['excel_row']} → {mapping['image']} row {mapping['row']}")
        print(f"   {mapping['png_category']} → '{mapping['label']}'")
    
    # Save summary report
    summary_path = train_dir / "mapping_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EXCEL TO PNG MAPPING SUMMARY - {extraction_suffix.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Excel file: {excel_path}\n")
        f.write(f"Excel rows: {len(df_excel)}\n")
        f.write(f"Extraction folder: {extracted_dir}\n")
        f.write(f"Image folders: {len(image_folders)}\n")
        f.write(f"Total mappings: {len(all_mappings)}\n\n")
        
        f.write("MAPPINGS BY CATEGORY:\n")
        f.write("-"*30 + "\n")
        for cat, count in sorted(category_counts.items()):
            f.write(f"{cat}: {count}\n")
    
    print(f"\n📝 Summary saved to: {summary_path}")
    
    return all_mappings

def check_excel_columns():
    """Check what's actually in the Excel file"""
    
    excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
    if not excel_path.exists():
        print("❌ Excel file not found!")
        return
    
    df = pd.read_csv(excel_path)
    
    print("📊 EXCEL FILE ANALYSIS")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    print(f"\n📋 COLUMN SAMPLES (first 5 rows):")
    for col in df.columns:
        sample_values = df[col].head(5).tolist()
        non_empty = [str(v) for v in sample_values if pd.notna(v) and str(v).strip()]
        if non_empty:
            print(f"\n{col}:")
            for i, val in enumerate(non_empty[:3]):  # Show first 3 non-empty
                print(f"  Row {i}: '{val}'")
    
    print(f"\n🎯 WHAT WE EXTRACT FROM IMAGES:")
    # Check what extraction folders exist
    extraction_patterns = ["extracted_cells", "extracted_cells_aligned", "extracted_cells_original"]
    
    for pattern in extraction_patterns:
        extracted_dirs = list(Path("data").glob(f"{pattern}*"))
        for extracted_dir in extracted_dirs:
            if extracted_dir.is_dir():
                print(f"\nFolder: {extracted_dir.name}")
                # Check first folder inside
                subfolders = list(extracted_dir.iterdir())
                if subfolders:
                    first_folder = subfolders[0]
                    head_rows_dir = first_folder / "head_rows"
                    if head_rows_dir.exists():
                        png_files = list(head_rows_dir.glob("HEAD_row00_*.png"))
                        categories = sorted(list(set([
                            f.stem.split('_')[2] for f in png_files
                        ])))
                        print(f"  Extracted categories: {categories}")
    
    print(f"\n" + "="*60)
    print("💡 SUGGESTED MAPPING STRATEGY:")
    print("1. Look at Excel columns above")
    print("2. Match them to extracted categories")
    print("3. Update the 'column_mapping' dictionary in the script")
    print("\nExample mappings (adjust based on your Excel):")
    print("  - 'Race' column in Excel → 'race' PNGs")
    print("  - 'House_Number' column → 'house_number' PNGs")
    print("  - 'Owned_Home_Value' column → 'price_rent' PNGs (or 'rented_owned')")

def quick_map_test(extraction_suffix="aligned"):
    """Quick test mapping first few rows"""
    
    excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
    if not excel_path.exists():
        print("❌ Excel file not found!")
        return
    
    df_excel = pd.read_csv(excel_path)
    
    extracted_dir = Path(f"data/extracted_cells_{extraction_suffix}")
    if not extracted_dir.exists():
        print(f"❌ Extracted directory not found: {extracted_dir}")
        return
    
    image_folders = sorted([f for f in extracted_dir.iterdir() if f.is_dir()])
    
    if not image_folders:
        print(f"❌ No image folders in {extracted_dir}")
        return
    
    print("🔍 QUICK MAPPING TEST")
    print("="*60)
    print(f"Using extraction folder: {extraction_suffix}")
    
    # Test with first 5 Excel rows
    for i in range(min(5, len(df_excel))):
        print(f"\n📝 Excel Row {i}:")
        row = df_excel.iloc[i]
        
        # Which image folder would this be? (back to front)
        # Excel row 0 = last image, last row
        if image_folders:
            last_image = image_folders[-1]  # Last image
            head_rows_dir = last_image / "head_rows"
            
            if head_rows_dir.exists():
                # Last row in image is typically row 39 (for 40 rows, 0-39)
                last_row = 39
                
                print(f"  Would map to: {last_image.name}, row {last_row}")
                
                # Show available Excel data
                for col in ['Race', 'House_Number', 'Street_Name', 'Owned_Home_Value', 'Rented']:
                    if col in df_excel.columns:
                        value = row.get(col, 'N/A')
                        if pd.notna(value) and str(value).strip():
                            print(f"  {col}: '{value}'")
                
                # Check if PNG files exist
                print(f"  Checking PNGs for row {last_row}:")
                for category in ['race', 'house_number', 'rented_owned']:
                    png_file = head_rows_dir / f"HEAD_row{last_row:02d}_{category}.png"
                    exists = "✓" if png_file.exists() else "✗"
                    print(f"    {category}.png: {exists}")

def analyze_mapping_coverage():
    """Analyze how much of the Excel data we can map"""
    
    excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
    if not excel_path.exists():
        print("❌ Excel file not found!")
        return
    
    df_excel = pd.read_csv(excel_path)
    
    print("📊 MAPPING COVERAGE ANALYSIS")
    print("="*60)
    
    # Check all extraction folders
    extraction_folders = []
    for folder in Path("data").glob("extracted_cells_*"):
        if folder.is_dir():
            extraction_folders.append(folder)
    
    if not extraction_folders:
        print("No extraction folders found!")
        return
    
    print(f"Excel rows: {len(df_excel)}")
    print(f"Extraction folders found: {len(extraction_folders)}")
    
    for extracted_dir in extraction_folders:
        print(f"\n📁 {extracted_dir.name}:")
        
        image_folders = sorted([f for f in extracted_dir.iterdir() if f.is_dir()])
        if not image_folders:
            print("  No image folders")
            continue
        
        total_head_rows = 0
        for img_folder in image_folders:
            head_rows_dir = img_folder / "head_rows"
            if head_rows_dir.exists():
                race_files = list(head_rows_dir.glob("HEAD_row*_race.png"))
                row_numbers = set([
                    int(f.stem.split('_')[1].replace('row', '')) 
                    for f in race_files
                ])
                total_head_rows += len(row_numbers)
        
        print(f"  Images: {len(image_folders)}")
        print(f"  Total head rows: {total_head_rows}")
        print(f"  Excel coverage: {min(total_head_rows, len(df_excel))}/{len(df_excel)} rows")
        
        if total_head_rows >= len(df_excel):
            print(f"  ✅ Sufficient head rows for Excel data")
        else:
            print(f"  ⚠️  Missing {len(df_excel) - total_head_rows} head rows")

if __name__ == "__main__":
    print("="*70)
    print("📊 CENSUS DATA MAPPING: PNGs → EXCEL")
    print("="*70)
    
    print("\nChoose action:")
    print("1. Check Excel columns (what data we have)")
    print("2. Quick mapping test with aligned images")
    print("3. Map ALL extracted PNGs to Excel (with choice of extraction)")
    print("4. Analyze mapping coverage")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        check_excel_columns()
    elif choice == "2":
        print("\nChoose extraction source:")
        print("1. Aligned images (deskewed)")
        print("2. Original images")
        print("3. Ready images")
        source_choice = input("Enter choice (1-3): ").strip()
        
        if source_choice == "1":
            suffix = "aligned"
        elif source_choice == "2":
            suffix = "original"
        elif source_choice == "3":
            suffix = "ready"
        else:
            print("Using aligned images by default")
            suffix = "aligned"
            
        quick_map_test(suffix)
    elif choice == "3":
        print("\n⚠️  IMPORTANT: Check your Excel columns first if you haven't!")
        print("Run option 1 to see what columns are in your Excel")
        
        excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
        if excel_path.exists():
            df = pd.read_csv(excel_path)
            print(f"\nCurrent Excel has {len(df)} rows, columns: {list(df.columns)[:10]}...")
            
            print(f"\nChoose extraction source:")
            print("1. Aligned images (deskewed) - RECOMMENDED")
            print("2. Original images")
            print("3. Ready images")
            print("4. Custom folder name")
            source_choice = input("Enter choice (1-4): ").strip()
            
            if source_choice == "1":
                extraction_suffix = "aligned"
            elif source_choice == "2":
                extraction_suffix = "original"
            elif source_choice == "3":
                extraction_suffix = "ready"
            elif source_choice == "4":
                extraction_suffix = input("Enter custom suffix (e.g., 'test' for 'extracted_cells_test'): ").strip()
            else:
                print("Using aligned images by default")
                extraction_suffix = "aligned"
            
            confirm = input(f"\nMap using {extraction_suffix} images? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                mappings = map_extracted_to_excel(extraction_suffix)
                
                if mappings:
                    print(f"\n🎯 NEXT STEPS:")
                    print(f"1. Check training dataset: data/training_dataset_{extraction_suffix}/")
                    print(f"2. Verify labels look correct")
                    print(f"3. Prepare for HPC training with labeled data")
        else:
            print("❌ Excel file not found!")
    elif choice == "4":
        analyze_mapping_coverage()
    else:
        print("Invalid choice.")