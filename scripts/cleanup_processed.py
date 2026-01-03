import os
import glob

def cleanup_processed_folder():
    """Clean up the processed folder to keep only essential files"""
    
    processed_dir = 'data/processed'
    
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} does not exist")
        return
    
    print("=== CLEANING UP PROCESSED FOLDER ===")
    
    # List of ESSENTIAL files to KEEP
    essential_files = [
        'preprocessed_image.png',
        'table_structure.png', 
        'smart_adaptive_grid.png',
        'head_row_zoom.png',
        'smart_extraction_report.txt'
    ]
    
    # Get all files in processed folder
    all_files = glob.glob(os.path.join(processed_dir, '*'))
    
    print(f"Found {len(all_files)} files in {processed_dir}")
    
    # Keep only essential files
    kept_count = 0
    deleted_count = 0
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        if file_name in essential_files:
            print(f"✅ KEEPING: {file_name}")
            kept_count += 1
        else:
            try:
                os.remove(file_path)
                print(f"🗑️  DELETED: {file_name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Could not delete {file_name}: {e}")
    
    print(f"\n📊 CLEANUP COMPLETE:")
    print(f"   Kept: {kept_count} essential files")
    print(f"   Deleted: {deleted_count} unnecessary files")
    
    # Show what remains
    print(f"\n📁 CURRENT CONTENTS of {processed_dir}:")
    remaining_files = glob.glob(os.path.join(processed_dir, '*'))
    for file_path in remaining_files:
        file_name = os.path.basename(file_path)
        print(f"   - {file_name}")

if __name__ == "__main__":
    cleanup_processed_folder()