import os
import shutil
from pathlib import Path

print("CHECKING FOR OLD EXTRACTION FILES")
print("=" * 50)

extracted_dir = Path("data/extracted_cells")
backup_dir = Path("data/old_extraction_backup")

# Find old format files
old_files = list(extracted_dir.glob("cell_*.png"))

if not old_files:
    print("No old extraction files found.")
    print("Your extraction structure is clean.")
    exit(0)

print(f"Found {len(old_files)} old extraction files.")

# Ask for confirmation
response = input(f"Move {len(old_files)} files to backup? (y/n): ")
if response.lower() != 'y':
    print("Cleanup cancelled.")
    exit(0)

# Create backup directory
backup_dir.mkdir(exist_ok=True)

# Move files
moved_count = 0
for old_file in old_files:
    try:
        shutil.move(str(old_file), str(backup_dir / old_file.name))
        moved_count += 1
    except Exception as e:
        print(f"Error moving {old_file.name}: {e}")

print(f"Moved {moved_count} files to {backup_dir}")
print("")
print("NOW RUN: python scripts/batch_smart_extraction.py")
print("This will create the new folder structure.")
