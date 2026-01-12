"""
Cleanup Unnecessary Scripts
Safely removes obsolete/redundant scripts from the project.
"""

import os
import shutil
from pathlib import Path

# Scripts that are definitely safe to delete (obsolete/redundant)
SAFE_TO_DELETE = [
    "scripts/check_status.py",              # Empty file
    "scripts/process_all_images.py",         # Superseded by batch_smart_extraction.py
    "scripts/preprocess_bw.py",              # Integrated into other scripts
    "scripts/column_segmentation.py",        # Early development script
    "scripts/create_dataset.py",             # Superseded by create_complete_mapping.py
]

# Scripts to review (may be redundant but verify first)
REVIEW_NEEDED = [
    "scripts/force_read_excel.py",           # May be redundant with extract_clean_data.py
    "scripts/map_to_excel.py",               # May be redundant with create_complete_mapping.py
    "scripts/smart_adaptive_extraction.py",  # May be integrated into batch_smart_extraction.py
    "scripts/deskew_correct_direction.py",   # May overlap with calibrated_deskew.py
    "scripts/batch_extraction_deskewed.py", # May overlap with batch_smart_extraction.py
    "scripts/accurate_head_detector.py",    # May overlap with final_head_row_extractor.py
]

def check_file_exists(filepath):
    """Check if file exists and return info"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        return True, size
    return False, 0

def list_files_to_delete():
    """List files that are safe to delete"""
    print("="*70)
    print("🗑️  SCRIPTS SAFE TO DELETE (Obsolete/Redundant)")
    print("="*70)
    print()
    
    files_to_delete = []
    for filepath in SAFE_TO_DELETE:
        exists, size = check_file_exists(filepath)
        if exists:
            files_to_delete.append((filepath, size))
            print(f"  ✓ {filepath}")
            print(f"    Size: {size:,} bytes")
        else:
            print(f"  ✗ {filepath} (not found)")
        print()
    
    return files_to_delete

def list_files_to_review():
    """List files that need review before deletion"""
    print("="*70)
    print("🔍 SCRIPTS TO REVIEW (May be redundant)")
    print("="*70)
    print()
    
    for filepath in REVIEW_NEEDED:
        exists, size = check_file_exists(filepath)
        if exists:
            print(f"  ⚠️  {filepath}")
            print(f"     Size: {size:,} bytes")
            print(f"     Action: Review manually before deleting")
        else:
            print(f"  ✗ {filepath} (not found)")
        print()

def delete_files(files_to_delete, dry_run=True):
    """Delete files (with dry-run option)"""
    if not files_to_delete:
        print("No files to delete.")
        return
    
    print("="*70)
    if dry_run:
        print("🔍 DRY RUN - No files will be deleted")
    else:
        print("🗑️  DELETING FILES")
    print("="*70)
    print()
    
    deleted_count = 0
    failed_count = 0
    
    for filepath, size in files_to_delete:
        path = Path(filepath)
        
        if dry_run:
            print(f"  [DRY RUN] Would delete: {filepath}")
        else:
            try:
                path.unlink()
                print(f"  ✅ Deleted: {filepath}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {filepath}: {e}")
                failed_count += 1
    
    print()
    if dry_run:
        print(f"📊 DRY RUN SUMMARY:")
        print(f"   Would delete: {len(files_to_delete)} files")
    else:
        print(f"📊 DELETION SUMMARY:")
        print(f"   Successfully deleted: {deleted_count} files")
        print(f"   Failed: {failed_count} files")

def create_backup(files_to_delete):
    """Create backup of files before deletion"""
    backup_dir = Path("scripts/backup_deleted")
    backup_dir.mkdir(exist_ok=True)
    
    print(f"📦 Creating backup in: {backup_dir}")
    print()
    
    backed_up = 0
    for filepath, _ in files_to_delete:
        source = Path(filepath)
        if source.exists():
            dest = backup_dir / source.name
            try:
                shutil.copy2(source, dest)
                print(f"  ✅ Backed up: {source.name}")
                backed_up += 1
            except Exception as e:
                print(f"  ❌ Failed to backup {source.name}: {e}")
    
    print()
    print(f"📊 Backup complete: {backed_up} files backed up")
    return backup_dir

def main():
    """Main cleanup function"""
    print("="*70)
    print("🧹 SCRIPT CLEANUP UTILITY")
    print("="*70)
    print()
    
    # List files safe to delete
    files_to_delete = list_files_to_delete()
    
    print()
    
    # List files to review
    list_files_to_review()
    
    print()
    print("="*70)
    print("OPTIONS")
    print("="*70)
    print()
    print("1. Show files to delete (already shown above)")
    print("2. Create backup of files before deletion")
    print("3. Delete files (DRY RUN - safe, shows what would be deleted)")
    print("4. Delete files (ACTUAL - permanently removes files)")
    print("5. Exit")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        list_files_to_delete()
        list_files_to_review()
    
    elif choice == "2":
        if files_to_delete:
            backup_dir = create_backup(files_to_delete)
            print(f"\n✅ Backup created at: {backup_dir}")
        else:
            print("No files to backup.")
    
    elif choice == "3":
        delete_files(files_to_delete, dry_run=True)
    
    elif choice == "4":
        if not files_to_delete:
            print("No files to delete.")
            return
        
        print("\n⚠️  WARNING: This will permanently delete files!")
        confirm = input("Type 'DELETE' to confirm: ").strip()
        
        if confirm == "DELETE":
            # Create backup first
            print("\nCreating backup first...")
            create_backup(files_to_delete)
            print()
            
            # Delete files
            delete_files(files_to_delete, dry_run=False)
        else:
            print("Deletion cancelled.")
    
    elif choice == "5":
        print("Exiting...")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()











