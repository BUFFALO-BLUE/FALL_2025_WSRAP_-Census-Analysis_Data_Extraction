"""
COMPLETE MAPPING: Extracted Cell Images → Excel Transcription Data
This creates the FINAL training dataset for machine learning.
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import json
import sys
import subprocess

# ============================================================================
# HANDLE MISSING DEPENDENCIES
# ============================================================================
def install_missing_package(package_name):
    """Install missing package using pip"""
    print(f"📦 Installing missing package: {package_name}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to install {package_name}: {e}")
        print(f"   Please run manually: pip install {package_name}")
        return False

# Try to import tqdm, install if missing
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  Module 'tqdm' not found. Attempting to install...")
    if install_missing_package("tqdm"):
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
            print("✅ tqdm imported successfully after installation")
        except ImportError:
            TQDM_AVAILABLE = False
            print("⚠️  Could not import tqdm even after installation attempt")
    else:
        TQDM_AVAILABLE = False

# Try to import cv2, install if missing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  Module 'cv2' (OpenCV) not found. Attempting to install...")
    if install_missing_package("opencv-python"):
        try:
            import cv2
            CV2_AVAILABLE = True
            print("✅ cv2 imported successfully after installation")
        except ImportError:
            CV2_AVAILABLE = False
            print("⚠️  Could not import cv2 even after installation attempt")
    else:
        CV2_AVAILABLE = False

# Create a simple tqdm replacement if not available
if not TQDM_AVAILABLE:
    print("⚠️  Using simple progress display (tqdm not available)")
    class SimpleTqdm:
        def __init__(self, iterable, desc="", **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
        
        def __iter__(self):
            for i, item in enumerate(self.iterable):
                if self.total and (i % max(1, self.total // 10) == 0):
                    print(f"   {self.desc}: {i}/{self.total if self.total else '?'}")
                yield item
    
    tqdm = SimpleTqdm

def create_complete_mapping():
    """
    MASTER FUNCTION: Maps ALL extracted cells to ALL Excel transcriptions
    
    This is the CRITICAL step that connects:
    • Extracted PNG cell images (handwriting)
    • Excel transcriptions (labels for training)
    
    Creates a complete training dataset for supervised ML.
    """
    
    print("="*80)
    print("📊 CREATING COMPLETE CELL-TO-EXCEL MAPPING")
    print("="*80)
    
    # ============================================================================
    # STEP 1: LOAD AND VALIDATE DATA
    # ============================================================================
    
    print("\n1. 📂 LOADING DATA SOURCES")
    print("-"*40)
    
    # A. Load cleaned Excel data (our labels/answers)
    excel_path = Path("data/from_jeremy/transcriptions/clean_census_data.csv")
    if not excel_path.exists():
        print(f"❌ Missing cleaned Excel data: {excel_path}")
        print("   Run first: python scripts/extract_clean_data.py")
        return None, None
    
    df_excel = pd.read_csv(excel_path)
    print(f"✅ Excel data loaded: {len(df_excel)} rows")
    print(f"   Columns: {list(df_excel.columns)}")
    
    # Show sample of Excel data
    print(f"\n🔍 Excel data sample (first 3 rows):")
    print(df_excel.head(3).to_string())
    
    # B. Check extracted cells structure
    extracted_root = Path("data/extracted_cells")
    if not extracted_root.exists():
        print(f"❌ No extracted cells found: {extracted_root}")
        print("   Run first: python scripts/batch_smart_extraction.py")
        return None, None
    
    # Find all image folders (should be named like m-t0627-00538-00634)
    image_folders = []
    for item in extracted_root.iterdir():
        if item.is_dir() and item.name.startswith('m-t0627-'):
            image_folders.append(item)
    
    print(f"✅ Found {len(image_folders)} extracted image folders")
    
    if len(image_folders) == 0:
        print("❌ No valid image folders found!")
        print("   Check if batch_smart_extraction.py created proper structure")
        return None, None
    
    # Check first folder structure
    sample_folder = image_folders[0]
    print(f"📁 Sample folder: {sample_folder.name}")
    
    head_rows = list((sample_folder / "head_rows").glob("*.png")) if (sample_folder / "head_rows").exists() else []
    non_head_rows = list((sample_folder / "non_head_rows").glob("*.png")) if (sample_folder / "non_head_rows").exists() else []
    
    print(f"   Head rows: {len(head_rows)} PNGs")
    print(f"   Non-head rows: {len(non_head_rows)} PNGs")
    
    if len(head_rows) == 0:
        print("❌ No cell images found in sample folder!")
        print("   Extraction may have failed")
        return None, None
    
    # ============================================================================
    # STEP 2: UNDERSTAND THE MAPPING LOGIC
    # ============================================================================
    
    print("\n2. 🧠 UNDERSTANDING THE MAPPING STRATEGY")
    print("-"*40)
    
    """
    CRITICAL ASSUMPTION (from Jeremy's response):
    Jeremy said: "yeah I started back to front and looking at it right now it follows that order"
    
    This means Excel rows are organized in the SAME ORDER as census images.
    
    EXAMPLE:
    • Excel rows 0-39 → Image 1 (m-t0627-00538-00634)
    • Excel rows 40-79 → Image 2 (m-t0627-00538-00635)
    • Excel rows 80-119 → Image 3 (m-t0627-00538-00636)
    • etc.
    
    Each image has 40 rows of data (0-39 in census form).
    Each Excel row corresponds to one census form row.
    """
    
    print("📐 Mapping Strategy:")
    print("   Excel Row 0-39   → Image 1, Rows 0-39")
    print("   Excel Row 40-79  → Image 2, Rows 0-39")
    print("   Excel Row 80-119 → Image 3, Rows 0-39")
    print("   ...")
    
    # ============================================================================
    # STEP 3: CREATE THE MAPPING
    # ============================================================================
    
    print("\n3. 🔗 CREATING CELL-TO-EXCEL MAPPING")
    print("-"*40)
    
    # Define which fields we're extracting
    # These MUST match your column names in smart_adaptive_extraction.py
    FIELDS_TO_MAP = {
        'race': 'Race',
        'house_number': 'House_Number',
        'street': 'Street_Name',
        'rented_owned': 'Rented_or_Owned',  # We'll derive this
        'price_rent': 'Value'  # Combined field
    }
    
    all_mappings = []  # Will store EVERY mapping
    stats = {
        'total_cells_processed': 0,
        'cells_with_labels': 0,
        'cells_without_labels': 0,
        'images_processed': 0,
        'excel_rows_used': 0
    }
    
    # Sort image folders to ensure consistent order
    image_folders.sort(key=lambda x: x.name)
    
    print(f"🔍 Processing {len(image_folders)} image folders...")
    
    # Process each image folder
    for image_idx, image_folder in enumerate(tqdm(image_folders, desc="Mapping images")):
        image_name = image_folder.name  # e.g., "m-t0627-00538-00634"
        
        # Calculate which Excel rows correspond to this image
        # Image 1: rows 0-39, Image 2: rows 40-79, etc.
        excel_start_row = image_idx * 40
        excel_end_row = excel_start_row + 40
        
        # Check if we have enough Excel rows
        if excel_end_row > len(df_excel):
            print(f"\n⚠️  Warning: Image {image_name} needs rows {excel_start_row}-{excel_end_row}")
            print(f"   But Excel only has {len(df_excel)} rows")
            print(f"   Stopping - processed {image_idx} images out of {len(image_folders)}")
            break
        
        # Get Excel data for this image (40 rows)
        image_excel_data = df_excel.iloc[excel_start_row:excel_end_row].copy()
        stats['excel_rows_used'] += len(image_excel_data)
        
        # Process head rows for this image
        head_rows_dir = image_folder / "head_rows"
        if head_rows_dir.exists():
            head_cells = list(head_rows_dir.glob("HEAD_*.png"))
            
            for cell_path in head_cells:
                # Parse filename: HEAD_row00_race.png
                filename = cell_path.stem  # "HEAD_row00_race"
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    try:
                        row_num = int(parts[1].replace('row', ''))  # 0, 1, 2, ..., 39
                    except ValueError:
                        print(f"⚠️  Could not parse row number from: {filename}")
                        continue
                    
                    field = parts[2]  # 'race', 'house_number', etc.
                    
                    # Check if this is a field we want to map
                    if field in FIELDS_TO_MAP:
                        stats['total_cells_processed'] += 1
                        
                        # Get the corresponding Excel row for this census row
                        if row_num < len(image_excel_data):
                            excel_row = image_excel_data.iloc[row_num]
                            excel_column = FIELDS_TO_MAP[field]
                            
                            # Get the label from Excel
                            if excel_column in excel_row:
                                label = excel_row[excel_column]
                                
                                # Special handling for combined fields
                                if field == 'rented_owned':
                                    # Determine if rented or owned
                                    owned_val = excel_row.get('Owned_Home_Value', '')
                                    rented_val = excel_row.get('Rented', '')
                                    
                                    if pd.notna(owned_val) and str(owned_val).strip():
                                        label = 'Owned'
                                    elif pd.notna(rented_val) and str(rented_val).strip():
                                        label = 'Rented'
                                    else:
                                        label = ''
                                
                                elif field == 'price_rent':
                                    # Get either owned value or rent
                                    owned_val = excel_row.get('Owned_Home_Value', '')
                                    rented_val = excel_row.get('Rented', '')
                                    
                                    if pd.notna(owned_val) and str(owned_val).strip():
                                        label = owned_val
                                    elif pd.notna(rented_val) and str(rented_val).strip():
                                        # Extract just the number from "Rented at $35"
                                        if isinstance(rented_val, str) and '$' in rented_val:
                                            label = rented_val.split('$')[-1].strip()
                                        else:
                                            label = rented_val
                                    else:
                                        label = ''
                                
                                # Only map if we have a valid label
                                if pd.notna(label) and str(label).strip() != '':
                                    # Create mapping entry
                                    mapping_entry = {
                                        'cell_image': cell_path.name,
                                        'cell_path': str(cell_path),
                                        'census_image': image_name,
                                        'row_in_form': row_num,
                                        'field_type': field,
                                        'excel_label': str(label).strip(),
                                        'excel_row_index': excel_start_row + row_num,
                                        'excel_column': excel_column,
                                        'mapping_status': 'MAPPED'
                                    }
                                    
                                    all_mappings.append(mapping_entry)
                                    stats['cells_with_labels'] += 1
                                else:
                                    stats['cells_without_labels'] += 1
        
        stats['images_processed'] += 1
        
        # Show progress every 10 images
        if (image_idx + 1) % 10 == 0:
            print(f"   Processed {image_idx + 1}/{len(image_folders)} images")
            print(f"   Mappings created: {stats['cells_with_labels']}")
    
    # ============================================================================
    # STEP 4: SAVE THE MAPPING RESULTS
    # ============================================================================
    
    print("\n4. 💾 SAVING MAPPING RESULTS")
    print("-"*40)
    
    if len(all_mappings) == 0:
        print("❌ No mappings created!")
        print("   Check if field names match between extraction and Excel")
        return None, None
    
    # Create DataFrames for different formats
    df_mappings = pd.DataFrame(all_mappings)
    
    # Ensure output directory exists
    output_dir = Path("data/extracted_cells")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Save complete mapping as CSV
    mapping_csv_path = output_dir / "complete_mapping.csv"
    df_mappings.to_csv(mapping_csv_path, index=False)
    print(f"✅ Complete mapping saved: {mapping_csv_path}")
    print(f"   Total mappings: {len(df_mappings)}")
    
    # B. Save organized by field type
    for field in FIELDS_TO_MAP.keys():
        field_mappings = df_mappings[df_mappings['field_type'] == field]
        if len(field_mappings) > 0:
            field_path = output_dir / f"mapping_{field}.csv"
            field_mappings.to_csv(field_path, index=False)
            print(f"   {field}: {len(field_mappings)} mappings")
    
    # C. Save statistics
    stats['mapping_percentage'] = (stats['cells_with_labels'] / stats['total_cells_processed'] * 100) if stats['total_cells_processed'] > 0 else 0
    
    stats_report = f"""COMPLETE MAPPING STATISTICS
==============================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SOURCES:
-------------
Excel rows available: {len(df_excel)}
Image folders found: {len(image_folders)}
Expected cells per image: ~200 (head rows only)

PROCESSING RESULTS:
-------------------
Images processed: {stats['images_processed']}
Excel rows used: {stats['excel_rows_used']}

CELL PROCESSING:
----------------
Total cells processed: {stats['total_cells_processed']}
Cells successfully mapped: {stats['cells_with_labels']}
Cells without labels: {stats['cells_without_labels']}
Mapping success rate: {stats['mapping_percentage']:.1f}%

FIELD DISTRIBUTION:
-------------------"""
    
    for field in FIELDS_TO_MAP.keys():
        count = len(df_mappings[df_mappings['field_type'] == field])
        stats_report += f"\n{field}: {count} mappings"
    
    stats_report += f"""

NEXT STEPS:
-----------
1. Review complete_mapping.csv for accuracy
2. Create training dataset from mapped cells
3. Train machine learning model
4. Validate with Jeremy's QA/QC

NOTES:
------
- Mapping assumes Excel rows 0-39 = Image 1 rows 0-39
- Each subsequent 40 Excel rows = next image
- Only head rows are mapped (most important data)
- Fields mapped: {', '.join(FIELDS_TO_MAP.keys())}
"""
    
    stats_path = output_dir / "mapping_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write(stats_report)
    
    print(f"✅ Statistics saved: {stats_path}")
    
    # D. Create sample preview
    print(f"\n🔍 SAMPLE MAPPINGS (first 10):")
    print("-"*60)
    if len(df_mappings) >= 10:
        sample = df_mappings.head(10)[['cell_image', 'census_image', 'row_in_form', 'field_type', 'excel_label']]
        print(sample.to_string(index=False))
    else:
        print(df_mappings[['cell_image', 'census_image', 'row_in_form', 'field_type', 'excel_label']].to_string(index=False))
    
    # ============================================================================
    # STEP 5: CREATE TRAINING DATASET FOLDER STRUCTURE
    # ============================================================================
    
    print("\n5. 🗂️ CREATING TRAINING DATASET STRUCTURE")
    print("-"*40)
    
    training_dir = Path("data/training_dataset")
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Create organized folder structure
    for field in FIELDS_TO_MAP.keys():
        (training_dir / field).mkdir(exist_ok=True)
    
    # Copy mapped images to organized structure
    print(f"📁 Organizing {len(df_mappings)} mapped cells...")
    
    copied_count = 0
    for idx, mapping in df_mappings.iterrows():
        source_path = Path(mapping['cell_path'])
        field = mapping['field_type']
        label = mapping['excel_label']
        
        if source_path.exists():
            # Create clean filename
            # Remove any characters that might cause issues in filenames
            clean_label = "".join(c for c in str(label) if c.isalnum() or c in " _-").rstrip()
            # Truncate very long labels
            if len(clean_label) > 50:
                clean_label = clean_label[:50]
            dest_filename = f"{mapping['census_image']}_row{mapping['row_in_form']:02d}_{field}_{clean_label}.png"
            dest_path = training_dir / field / dest_filename
            
            # Copy image
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"⚠️  Could not copy {source_path}: {e}")
        
        # Show progress every 100 copies
        if copied_count % 100 == 0 and copied_count > 0:
            print(f"   Copied {copied_count} images...")
    
    print(f"✅ Training dataset created: {training_dir}")
    print(f"   Images copied: {copied_count}")
    
    # Create dataset manifest
    manifest = {
        'creation_date': pd.Timestamp.now().isoformat(),
        'total_samples': copied_count,
        'fields': list(FIELDS_TO_MAP.keys()),
        'source_images': len(image_folders),
        'source_excel_rows': len(df_excel),
        'mapping_strategy': 'sequential_40_rows_per_image',
        'notes': 'Assumes Excel rows 0-39 = Image 1, rows 40-79 = Image 2, etc.'
    }
    
    manifest_path = training_dir / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Dataset manifest: {manifest_path}")
    
    # ============================================================================
    # STEP 6: CREATE SUMMARY FOR JEREMY/PROFESSOR
    # ============================================================================
    
    print("\n6. 📊 CREATING PROJECT SUMMARY")
    print("-"*40)
    
    summary = f"""CENSUS OCR PROJECT - COMPLETE MAPPING SUMMARY
==================================================

PROJECT STATUS:
---------------
✅ Data extraction: COMPLETE ({len(image_folders)} images)
✅ Excel cleaning: COMPLETE ({len(df_excel)} rows)
✅ Cell-to-Excel mapping: COMPLETE ({len(df_mappings)} mappings)
✅ Training dataset: CREATED ({copied_count} labeled samples)

MAPPING DETAILS:
----------------
Mapping strategy: Sequential 40-row blocks
- Excel rows 0-39 → {image_folders[0].name if image_folders else 'N/A'} (rows 0-39)
- Excel rows 40-79 → {image_folders[1].name if len(image_folders) > 1 else 'N/A'} (rows 0-39)
- etc.

Fields successfully mapped:
{chr(10).join([f'  • {field}: {len(df_mappings[df_mappings["field_type"]==field])} samples' for field in FIELDS_TO_MAP.keys()])}

DATA QUALITY:
-------------
Total cells available: {stats['total_cells_processed']}
Successfully mapped: {stats['cells_with_labels']} ({stats['mapping_percentage']:.1f}%)
Missing labels: {stats['cells_without_labels']}

NEXT STEPS FOR ML TRAINING:
---------------------------
1. Dataset ready at: {training_dir}/
2. Contains {copied_count} labeled handwriting samples
3. Organized by field type for training
4. Ready for HPC model training

FILES CREATED:
--------------
1. complete_mapping.csv - Complete cell-to-Excel mapping
2. mapping_*.csv - Field-specific mappings
3. mapping_statistics.txt - Detailed statistics
4. data/training_dataset/ - Organized training data
5. dataset_manifest.json - Dataset metadata

VALIDATION NEEDED:
------------------
Please verify:
1. Excel rows 0-39 correctly match {image_folders[0].name if image_folders else 'first image'}
2. Field mappings look correct (see sample above)
3. Labels match the handwriting in corresponding images

READY FOR MACHINE LEARNING TRAINING!
"""
    
    summary_path = output_dir / "project_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"✅ Project summary: {summary_path}")
    
    print("\n" + "="*80)
    print("🎉 COMPLETE MAPPING FINISHED!")
    print("="*80)
    
    print(f"""
✅ WHAT WAS ACCOMPLISHED:

1. MAPPED {len(df_mappings)} CELLS TO EXCEL DATA
   • Connected handwriting images to transcriptions
   • Applied sequential mapping strategy
   • Organized by field type

2. CREATED TRAINING DATASET
   • Location: {training_dir}/
   • Samples: {copied_count} labeled images
   • Ready for machine learning

3. GENERATED COMPLETE DOCUMENTATION
   • Statistics, manifests, summaries
   • Everything needed for Friday report

📊 KEY METRICS:
   • Images processed: {stats['images_processed']}
   • Excel rows used: {stats['excel_rows_used']}
   • Mappings created: {len(df_mappings)}
   • Success rate: {stats['mapping_percentage']:.1f}%

🚀 NEXT STEPS:

1. REVIEW THE MAPPING:
   Check: data/extracted_cells/complete_mapping.csv
   Verify first few rows match your expectations

2. SAMPLE CHECK:
   Pick 5-10 mappings and verify:
   • Does the PNG show readable handwriting?
   • Does the Excel label match what you see?

3. PREPARE FOR HPC:
   Dataset is ready for model training!
   Use: data/training_dataset/

4. FRIDAY REPORT:
   You now have concrete numbers:
   • {len(df_mappings)} mapped samples
   • {stats['mapping_percentage']:.1f}% success rate
   • Complete training dataset created

🎯 YOU HAVE SUCCESSFULLY BUILT A SUPERVISED LEARNING DATASET!
   This is the foundation for training your OCR model.
""")
    
    return df_mappings, training_dir

def verify_mapping_quality():
    """Help verify that mappings are correct"""
    
    print("\n" + "="*80)
    print("🔍 MAPPING QUALITY VERIFICATION")
    print("="*80)
    
    mapping_path = Path("data/extracted_cells/complete_mapping.csv")
    if not mapping_path.exists():
        print("❌ No mapping file found. Run create_complete_mapping() first.")
        return
    
    df = pd.read_csv(mapping_path)
    
    print("\nQuick verification instructions:")
    print("-"*40)
    
    print("""
1. OPEN THESE FILES SIDE-BY-SIDE:
   • complete_mapping.csv (in Excel or text editor)
   • The corresponding PNG files
   • clean_census_data.csv (Excel transcriptions)

2. CHECK SAMPLE MAPPINGS:
   Look at the first 5 rows of complete_mapping.csv.
   For each row:
   a. Open the PNG file listed in 'cell_path'
   b. Read the handwriting in the image
   c. Compare to 'excel_label' column
   d. Do they match?

3. EXAMPLE CHECK:
   If mapping says:
     cell_image: HEAD_row00_race.png
     excel_label: White
   
   Open HEAD_row00_race.png - does it show "White" handwriting?
   If yes → mapping is correct!
   If no → check Excel row or image filename

4. CHECK MAPPING STRATEGY:
   • Does Excel row 0 match Image 1, Row 0?
   • Does Excel row 40 match Image 2, Row 0?
   • This is what Jeremy confirmed "follows that order"
""")
    
    # Show sample for verification
    if len(df) > 0:
        print(f"\n🔍 SAMPLE FOR VERIFICATION (first 3 rows):")
        print("-"*60)
        sample = df.head(3)[['cell_image', 'census_image', 'row_in_form', 'field_type', 'excel_label', 'excel_row_index']]
        print(sample.to_string(index=False))
        
        print(f"\n📁 To verify, open these files:")
        for idx, row in df.head(3).iterrows():
            print(f"  {row['cell_image']} → Should show: '{row['excel_label']}'")
    else:
        print("⚠️  No mappings found in the file.")
    
    return df

def quick_status_check():
    """Check if prerequisites are met"""
    
    print("\n🔍 QUICK STATUS CHECK:")
    print("-"*40)
    
    checks = [
        ("Clean Excel data", Path("data/from_jeremy/transcriptions/clean_census_data.csv").exists()),
        ("Extracted cells folder", Path("data/extracted_cells").exists()),
        ("Extracted PNG files", any(Path("data/extracted_cells").glob("m-t0627-*/head_rows/*.png"))),
        ("Existing mapping", Path("data/extracted_cells/complete_mapping.csv").exists()),
    ]
    
    all_passed = True
    for check_name, check_passed in checks:
        status = "✅" if check_passed else "❌"
        print(f"{status} {check_name}")
        if not check_passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ Ready to run mapping or verification!")
    else:
        print("\n⚠️  Some prerequisites missing")
        print("\nRun these commands first:")
        print("1. python scripts/extract_clean_data.py")
        print("2. python scripts/batch_smart_extraction.py")

if __name__ == "__main__":
    print("COMPLETE CELL-TO-EXCEL MAPPING SYSTEM")
    print("="*80)
    
    print("\nChoose action:")
    print("1. Create complete mapping (main function)")
    print("2. Verify mapping quality")
    print("3. Quick status check")
    print("4. Install missing dependencies")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚨 This will map ALL extracted cells to ALL Excel data.")
        print("   It assumes: Excel rows 0-39 = Image 1, rows 40-79 = Image 2, etc.")
        print("   (This is what Jeremy confirmed)")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            result = create_complete_mapping()
            if result and result[0] is not None:
                mappings, training_dir = result
                print(f"\n🎉 MAPPING SUCCESSFUL!")
                print(f"   Training dataset ready at: {training_dir}")
                print(f"   Report for professor: data/extracted_cells/project_summary.txt")
            else:
                print("\n❌ Mapping failed. Check error messages above.")
        else:
            print("Mapping cancelled.")
    
    elif choice == "2":
        verify_mapping_quality()
    
    elif choice == "3":
        quick_status_check()
    
    elif choice == "4":
        print("\n📦 INSTALLING MISSING DEPENDENCIES")
        print("-"*40)
        install_missing_package("tqdm")
        install_missing_package("opencv-python")
        print("\n✅ Dependencies installation complete!")
        print("   Run the script again and choose option 1.")
    
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")