import pandas as pd
import os
import shutil
from pathlib import Path
import zipfile

def get_images_for_single_mapping():
    """Get the JPG images that correspond to your single_image_mapping.csv"""
    
    print("=" * 70)
    print("📸 GETTING JPGs FOR SINGLE IMAGE MAPPING")
    print("=" * 70)
    
    # 1. Read your mapping file
    mapping_path = Path("data/extracted_cells/single_image_mapping.csv")
    if not mapping_path.exists():
        print(f"❌ Mapping file not found: {mapping_path}")
        print("   Run: python scripts/create_single_image_csv.py first")
        return
    
    df_mapping = pd.read_csv(mapping_path)
    
    # Get unique image names from the mapping
    unique_images = df_mapping['image_name'].unique()
    print(f"📊 Found {len(unique_images)} unique images in mapping:")
    for img in unique_images:
        print(f"   • {img}")
    
    # 2. Find the original JPG files
    possible_sources = [
        Path("data/from_jeremy/images"),          # Original images
        Path("data/from_jeremy/images_deskewed"), # Deskewed images
        Path("data/from_jeremy/images_original")  # Backup location
    ]
    
    source_dir = None
    for source in possible_sources:
        if source.exists():
            # Check if at least one image exists here
            for img_name in unique_images:
                if (source / img_name).exists():
                    source_dir = source
                    print(f"✅ Found images in: {source_dir}")
                    break
            if source_dir:
                break
    
    if not source_dir:
        print("❌ Could not find any of the image directories!")
        print("   Please check if your images are in one of these locations:")
        for source in possible_sources:
            print(f"   - {source}")
        return
    
    # 3. Create a folder for the JPGs
    output_dir = Path("data/for_jeremy_review")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Copying JPGs to: {output_dir}")
    
    # 4. Copy each image
    copied_images = []
    missing_images = []
    
    for img_name in unique_images:
        source_path = source_dir / img_name
        
        if source_path.exists():
            dest_path = output_dir / img_name
            shutil.copy2(source_path, dest_path)
            copied_images.append(img_name)
            print(f"   ✅ Copied: {img_name}")
        else:
            missing_images.append(img_name)
            print(f"   ❌ Missing: {img_name}")
    
    # 5. Create a ZIP file
    zip_filename = "single_mapping_images.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_name in copied_images:
            img_path = output_dir / img_name
            zipf.write(img_path, img_name)
    
    print(f"\n✅ Created ZIP file: {zip_filename}")
    print(f"   Contains: {len(copied_images)} JPG images")
    
    if missing_images:
        print(f"\n⚠️  Missing {len(missing_images)} images:")
        for img in missing_images:
            print(f"   - {img}")
    
    # 6. Create a README file for Jeremy
    readme_content = f"""IMAGES FOR SINGLE MAPPING REVIEW
====================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Prepared by: Musarah Muhammad
Project: Census OCR Pipeline

CONTENTS:
---------
This ZIP contains {len(copied_images)} census JPG images that correspond to the
'single_image_mapping.csv' file sent to Professor Cohen.

IMAGES INCLUDED:
----------------
"""
    
    for i, img_name in enumerate(copied_images, 1):
        readme_content += f"{i:2d}. {img_name}\n"
    
    readme_content += f"""
CORRESPONDING DATA:
-------------------
These images are mapped to Excel transcriptions in:
- 'single_image_mapping.csv' (complete mapping for m-t0627-00538-00634.jpg)
- 'census_ocr_demonstration.xlsx' (Excel file with 4 sheets)

SPECIFIC MAPPING EXAMPLE:
-------------------------
Image: m-t0627-00538-00634.jpg
- Row 0: Race='White', House_Number='131', Street_Name='Chesire Street'
- Row 1: Race='White', House_Number='108', Street_Name='Ansonia'
- Row 2: Race='White', House_Number='517', Street_Name='New Britain Avenue'
- ... (40 total rows, 5 fields each)

INSTRUCTIONS FOR JEREMY:
------------------------
1. Open the JPG images in this folder
2. Compare the handwriting to the Excel transcriptions
3. Check accuracy for: Race, House Number, Street Name fields
4. Note any discrepancies or unclear handwriting

The goal is to establish baseline accuracy before scaling up.
"""
    
    readme_path = output_dir / "README_FOR_JEREMY.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n📄 Created README file: {readme_path}")
    
    # 7. Show statistics from the mapping
    print("\n📊 MAPPING STATISTICS:")
    print("-" * 40)
    
    # Count by field
    field_counts = df_mapping['field'].value_counts()
    for field, count in field_counts.items():
        print(f"   {field}: {count} cells")
    
    # Show sample of the mapping
    print(f"\n🔍 SAMPLE OF THE MAPPING (first 5 rows):")
    print(df_mapping.head().to_string(index=False))
    
    return zip_filename, copied_images

def create_targeted_email():
    """Create a targeted email for Jeremy focusing on the single mapping"""
    
    email_template = """
Hi Jeremy,

As requested, I'm sending you the specific census JPG images that correspond to the single-image mapping I shared with Professor Cohen.

Attached: single_mapping_images.zip

WHAT'S IN THE ZIP:
------------------
- The census JPG image(s) used in the 'single_image_mapping.csv' file
- A README file with specific instructions for your review

CONTEXT FOR YOUR REVIEW:
------------------------
You'll be comparing the *actual handwriting* in these JPGs against the *transcriptions in the Excel file* for this specific image.

For example, for image m-t0627-00538-00634.jpg:
- Row 0 should show: Race='White', House Number='131', Street='Chesire Street'
- Row 1 should show: Race='White', House Number='108', Street='Ansonia'
- etc.

This focused review will help us:
1. Verify the accuracy of my current mapping process
2. Identify any systematic errors in reading specific fields
3. Establish a baseline accuracy rate before scaling up

Please focus on these specific images and their corresponding Excel rows.

Let me know what accuracy rate you find!

Best,
Musarah
"""
    
    return email_template

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 1: GET SPECIFIC JPGs FROM MAPPING")
    print("=" * 70)
    
    zip_file, images = get_images_for_single_mapping()
    
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE EMAIL FOR JEREMY")
    print("=" * 70)
    
    email = create_targeted_email()
    print(email)
    
    print("\n" + "=" * 70)
    print("🎯 ACTION ITEMS")
    print("=" * 70)
    
    print(f"""
✅ WHAT TO DO NOW:

1. ATTACH THIS FILE TO EMAIL:
   - {zip_file}

2. USE THIS EMAIL TEMPLATE:
   Copy the email above and send to Jeremy

3. CONTINUE YOUR WORK TRACKING:
   python scripts/track_work.py
   (Start session → Transfer PNGs → End session → Log)

4. PREPARE FOR FRIDAY REPORT:
   You'll report:
   - How many PNGs you transferred
   - Your speed (PNGs/minute)
   - Jeremy's accuracy assessment

📊 YOU'RE ON TRACK:
   - Professor liked your Excel proof-of-concept ✓
   - You're getting specific images for review ✓
   - You're tracking your work speed ✓
   
   Friday's report will show concrete progress!
""")