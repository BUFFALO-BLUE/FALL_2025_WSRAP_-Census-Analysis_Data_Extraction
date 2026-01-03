import pandas as pd
import os
import shutil
from pathlib import Path
import zipfile
import json

def collect_cell_images_from_mapping():
    """Collect all extracted cell images mentioned in single_image_mapping.csv"""
    
    print("=" * 70)
    print("📦 COLLECTING EXTRACTED CELL IMAGES FOR REVIEW")
    print("=" * 70)
    
    # 1. Read your mapping file
    mapping_path = Path("data/extracted_cells/single_image_mapping.csv")
    if not mapping_path.exists():
        print(f"❌ Mapping file not found: {mapping_path}")
        print("   Run: python scripts/create_single_image_csv.py first")
        return None, None
    
    df_mapping = pd.read_csv(mapping_path)
    print(f"📊 Loaded mapping with {len(df_mapping)} entries")
    
    # 2. Parse the expected cell image filenames
    # The 'cell_image' column has format like: "HEAD_row00_race.png (expected)"
    # We need to extract just the filename: "HEAD_row00_race.png"
    expected_files = []
    
    for idx, row in df_mapping.iterrows():
        cell_image_field = str(row['cell_image'])
        
        # Extract the filename before the space and "(expected)"
        if '(' in cell_image_field:
            filename = cell_image_field.split('(')[0].strip()
        else:
            filename = cell_image_field
        
        # Clean it up
        if filename.endswith('.png'):
            expected_files.append({
                'expected_filename': filename,
                'image_name': row['image_name'],
                'row': row['row_in_form'],
                'field': row['field'],
                'excel_value': row['excel_value']
            })
    
    print(f"🔍 Looking for {len(expected_files)} expected cell images...")
    
    # 3. Search for these files in your extracted_cells structure
    # We need to check multiple possible locations
    
    possible_locations = [
        # New structure (from batch_smart_extraction.py)
        Path("data/extracted_cells") / "head_rows",
        Path("data/extracted_cells") / "non_head_rows",
        
        # Old structure (if you have it)
        Path("data/processed/sample_extraction"),
        Path("data/processed"),
        
        # Per-image folder structure (ideal)
        Path("data/extracted_cells/m-t0627-00538-00634/head_rows"),
        Path("data/extracted_cells/m-t0627-00538-00634/non_head_rows"),
    ]
    
    # Also check for any folder starting with m-t0627-
    extracted_root = Path("data/extracted_cells")
    if extracted_root.exists():
        for folder in extracted_root.iterdir():
            if folder.is_dir() and folder.name.startswith('m-t0627-'):
                possible_locations.append(folder / "head_rows")
                possible_locations.append(folder / "non_head_rows")
    
    # 4. Create output directory structure
    output_dir = Path("data/jeremy_review_cells")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "head_rows").mkdir(exist_ok=True)
    (output_dir / "non_head_rows").mkdir(exist_ok=True)
    (output_dir / "by_field").mkdir(exist_ok=True)
    
    # 5. Search and copy files
    found_files = []
    missing_files = []
    
    print("\n🔍 Searching for cell images...")
    
    for expected in expected_files:
        filename = expected['expected_filename']
        found = False
        
        # Search in all possible locations
        for location in possible_locations:
            if location.exists():
                file_path = location / filename
                if file_path.exists():
                    # Copy to organized structure
                    dest_dir = output_dir / "head_rows" if "HEAD" in filename else output_dir / "non_head_rows"
                    dest_path = dest_dir / filename
                    
                    # Also copy to field-based organization
                    field_dir = output_dir / "by_field" / expected['field']
                    field_dir.mkdir(exist_ok=True)
                    field_dest = field_dir / filename
                    
                    shutil.copy2(file_path, dest_path)
                    shutil.copy2(file_path, field_dest)
                    
                    found_files.append({
                        'filename': filename,
                        'source': str(file_path),
                        'image': expected['image_name'],
                        'row': expected['row'],
                        'field': expected['field'],
                        'excel_value': expected['excel_value']
                    })
                    
                    found = True
                    if len(found_files) % 20 == 0:
                        print(f"   Found {len(found_files)}/{len(expected_files)} files...")
                    break
        
        if not found:
            missing_files.append(expected)
    
    print(f"\n✅ Found {len(found_files)} cell images")
    print(f"⚠️  Missing {len(missing_files)} cell images")
    
    if len(found_files) == 0:
        print("\n❌ CRITICAL: No cell images found!")
        print("   Possible reasons:")
        print("   1. batch_smart_extraction.py hasn't been run")
        print("   2. Images are in a different location")
        print("   3. File naming is different than expected")
        return None, None
    
    # 6. Create a manifest file
    manifest_path = output_dir / "IMAGE_MANIFEST.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'extraction_date': pd.Timestamp.now().isoformat(),
            'total_expected': len(expected_files),
            'total_found': len(found_files),
            'missing_count': len(missing_files),
            'found_files': found_files,
            'missing_files': missing_files
        }, f, indent=2, default=str)
    
    # 7. Create a simple CSV manifest for easy viewing
    csv_manifest = []
    for file_info in found_files:
        csv_manifest.append({
            'cell_image': file_info['filename'],
            'census_image': file_info['image'],
            'row': file_info['row'],
            'field': file_info['field'],
            'excel_transcription': file_info['excel_value'],
            'status': 'FOUND'
        })
    
    for missing in missing_files:
        csv_manifest.append({
            'cell_image': missing['expected_filename'],
            'census_image': missing['image_name'],
            'row': missing['row'],
            'field': missing['field'],
            'excel_transcription': missing['excel_value'],
            'status': 'MISSING'
        })
    
    df_manifest = pd.DataFrame(csv_manifest)
    csv_path = output_dir / "cell_image_manifest.csv"
    df_manifest.to_csv(csv_path, index=False)
    
    # 8. Create ZIP file
    zip_filename = "extracted_cell_images_review.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all found images
        for file_info in found_files:
            filename = file_info['filename']
            src_path = output_dir / "head_rows" / filename if "HEAD" in filename else output_dir / "non_head_rows" / filename
            zipf.write(src_path, f"extracted_cells/{filename}")
        
        # Add manifest files
        zipf.write(manifest_path, "MANIFEST.json")
        zipf.write(csv_path, "cell_image_manifest.csv")
        
        # Add the original mapping for reference
        zipf.write(mapping_path, "original_mapping.csv")
    
    print(f"\n✅ Created ZIP file: {zip_filename}")
    print(f"   Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    print(f"   Contains: {len(found_files)} extracted cell images")
    
    # 9. Show what we found
    print("\n📋 SAMPLE OF FOUND IMAGES (first 10):")
    print("-" * 50)
    for i, file_info in enumerate(found_files[:10]):
        print(f"{i+1:2d}. {file_info['filename']}")
        print(f"    From: {file_info['image']}, Row {file_info['row']}, {file_info['field']}")
        print(f"    Excel: '{file_info['excel_value']}'")
    
    if missing_files:
        print(f"\n⚠️  MISSING FILES (first 10):")
        print("-" * 50)
        for i, missing in enumerate(missing_files[:10]):
            print(f"{i+1:2d}. {missing['expected_filename']}")
    
    return zip_filename, found_files

def diagnose_extraction_problem():
    """Help figure out why cell images might be missing"""
    
    print("\n" + "=" * 70)
    print("🔧 DIAGNOSING EXTRACTION ISSUES")
    print("=" * 70)
    
    # Check what's actually in extracted_cells
    extracted_root = Path("data/extracted_cells")
    
    if not extracted_root.exists():
        print("❌ data/extracted_cells/ doesn't exist!")
        print("   Run: python scripts/batch_smart_extraction.py")
        return
    
    print("📁 Current contents of data/extracted_cells/:")
    print("-" * 40)
    
    items = list(extracted_root.iterdir())
    if not items:
        print("   (empty directory)")
        return
    
    # Categorize items
    files = []
    folders = []
    
    for item in items:
        if item.is_file():
            files.append(item.name)
        else:
            folders.append(item.name)
    
    if files:
        print(f"📄 Files ({len(files)}):")
        for f in sorted(files)[:10]:  # Show first 10 files
            print(f"   • {f}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")
    
    if folders:
        print(f"\n📁 Folders ({len(folders)}):")
        for folder in sorted(folders)[:10]:
            folder_path = extracted_root / folder
            # Count PNGs in this folder
            png_count = 0
            if (folder_path / "head_rows").exists():
                png_count += len(list((folder_path / "head_rows").glob("*.png")))
            if (folder_path / "non_head_rows").exists():
                png_count += len(list((folder_path / "non_head_rows").glob("*.png")))
            
            print(f"   • {folder}/ ({png_count} PNGs)")
        
        # Check the first folder for structure
        if folders:
            sample_folder = extracted_root / folders[0]
            print(f"\n🔍 Sample folder structure: {sample_folder.name}/")
            if (sample_folder / "head_rows").exists():
                head_files = list((sample_folder / "head_rows").glob("*.png"))
                print(f"   head_rows/: {len(head_files)} PNGs")
                if head_files:
                    print(f"     Sample: {head_files[0].name}")
            
            if (sample_folder / "non_head_rows").exists():
                non_head_files = list((sample_folder / "non_head_rows").glob("*.png"))
                print(f"   non_head_rows/: {len(non_head_files)} PNGs")

def create_cell_review_email(found_count, zip_filename):
    """Create email for sending cell images to Jeremy"""
    
    email_template = f"""
Hi Jeremy,

As requested, I'm sending you the **extracted cell images** from my OCR pipeline for your QA/QC review.

Attached: {zip_filename}

WHAT'S IN THE ZIP:
------------------
- {found_count} extracted handwriting cell images (PNG files)
- Organized in folders by row type (head_rows/, non_head_rows/)
- MANIFEST.json with complete metadata
- cell_image_manifest.csv showing what each image should contain

CONTEXT FOR YOUR REVIEW:
------------------------
These are the actual *extracted cells* from the census forms - individual boxes containing:
1. Race information
2. House numbers  
3. Street names
4. Other census data fields

Each PNG file corresponds to a specific cell in the census table. For example:
- "HEAD_row00_race.png" = Race field from Row 0 of the census form
- "HEAD_row01_house_number.png" = House number from Row 1

WHAT TO CHECK:
--------------
1. **Extraction Accuracy**: Are the cells correctly cropped?
2. **Image Quality**: Is the handwriting legible in each PNG?
3. **Completeness**: Are all expected cells present?
4. **Alignment**: Do cells align with the Excel transcriptions?

This review will help us:
- Validate my image processing pipeline
- Identify any issues with cell extraction
- Establish confidence in the automated system

Please focus on the first 20-30 images to give me initial feedback on quality.

Best,
Musarah
"""
    
    return email_template

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 1: COLLECT EXTRACTED CELL IMAGES")
    print("=" * 70)
    
    zip_file, found_files = collect_cell_images_from_mapping()
    
    if zip_file and found_files:
        print("\n" + "=" * 70)
        print("STEP 2: CREATE EMAIL FOR JEREMY")
        print("=" * 70)
        
        email = create_cell_review_email(len(found_files), zip_file)
        print(email)
        
        print("\n" + "=" * 70)
        print("🎯 NEXT STEPS")
        print("=" * 70)
        
        print(f"""
✅ WHAT TO DO NOW:

1. ATTACH THIS FILE TO EMAIL:
   - {zip_file}

2. USE THIS EMAIL TEMPLATE:
   Copy the email above and send to Jeremy

3. CHECK YOUR EXTRACTION:
   If many files were missing, run:
   python scripts/batch_smart_extraction.py

4. PREPARE YOUR FRIDAY REPORT:
   You'll report:
   - Number of cell images extracted
   - Quality of extraction (from Jeremy's feedback)
   - Your PNG transfer speed

📊 YOU'RE SHOWING THE REAL WORK:
   This demonstrates your complete pipeline:
   Census JPGs → Cell Extraction → Organized PNGs → Quality Review
""")
    
    # Always run diagnosis to understand extraction status
    print("\n" + "=" * 70)
    print("STEP 3: CHECK EXTRACTION STATUS")
    print("=" * 70)
    
    diagnose_extraction_problem()
    
    # If extraction hasn't been run, provide clear instructions
    extracted_root = Path("data/extracted_cells")
    has_proper_folders = False
    
    if extracted_root.exists():
        for folder in extracted_root.iterdir():
            if folder.is_dir() and folder.name.startswith('m-t0627-'):
                has_proper_folders = True
                break
    
    if not has_proper_folders:
        print("\n🚨 ACTION REQUIRED: Run batch extraction!")
        print("-" * 40)
        print("To create the cell images Jeremy needs, run:")
        print("python scripts/batch_smart_extraction.py")
        print("\nThis will process all census images and create the PNG files.")