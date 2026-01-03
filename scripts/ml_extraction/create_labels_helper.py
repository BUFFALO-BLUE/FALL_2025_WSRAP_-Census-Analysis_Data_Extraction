"""
Helper script to create labels.csv for calibration samples
Makes it easier to label the calibration samples for training
"""

import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


def create_labels_template():
    """Create a template labels.csv file from calibration samples"""
    
    samples_dir = Path("data/calibration_samples")
    
    if not samples_dir.exists():
        print("❌ Calibration samples not found!")
        print("   Run: python scripts/accurate_head_detector.py (option 5)")
        return
    
    # Get all PNG files
    image_paths = sorted(list(samples_dir.glob("*.png")))
    
    if not image_paths:
        print("❌ No PNG files found")
        return
    
    print(f"📸 Found {len(image_paths)} calibration samples")
    
    # Check if labels.csv already exists
    labels_path = samples_dir / "labels.csv"
    
    if labels_path.exists():
        print(f"\n⚠️  labels.csv already exists: {labels_path}")
        response = input("   Overwrite? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("   Keeping existing file")
            return
    
    # Create template
    data = []
    for img_path in image_paths:
        data.append({
            'filename': img_path.name,
            'is_zero': '',  # Empty - user needs to fill
            'notes': ''  # Optional notes
        })
    
    df = pd.DataFrame(data)
    df.to_csv(labels_path, index=False)
    
    print(f"\n✅ Created template: {labels_path}")
    print(f"\n📝 NEXT STEPS:")
    print("="*60)
    print("1. Open labels.csv in Excel or text editor")
    print("2. For each row, set 'is_zero' to:")
    print("   - 1 if the image contains '0' (head of household)")
    print("   - 0 if it does NOT contain '0' (family member or empty)")
    print("3. Save the file")
    print("4. Run: python scripts/ml_extraction/train_zero_detector.py")
    print("\n💡 TIP: You can view images while labeling:")
    print("   - Open data/calibration_samples/ folder")
    print("   - View PNG files to see what each contains")


def view_samples_for_labeling():
    """View calibration samples to help with labeling"""
    
    samples_dir = Path("data/calibration_samples")
    
    if not samples_dir.exists():
        print("❌ Calibration samples not found!")
        return
    
    image_paths = sorted(list(samples_dir.glob("*.png")))
    
    if not image_paths:
        print("❌ No PNG files found")
        return
    
    print(f"📸 Viewing {len(image_paths)} samples")
    print("   Close window to view next image")
    print("   Press Ctrl+C to stop")
    
    try:
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path), 0)
            if img is not None:
                print(f"\n[{i+1}/{len(image_paths)}] {img_path.name}")
                print("   Does this contain '0'? (y/n/q to quit)")
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img, cmap='gray')
                plt.title(f"{img_path.name}\nDoes this contain '0'?", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show(block=False)
                
                # Wait a bit for display
                plt.pause(0.1)
                
                # Simple input (this won't work well with matplotlib, but gives idea)
                response = input("   Your answer (y/n/q): ").strip().lower()
                
                if response == 'q':
                    break
                
                plt.close()
    
    except KeyboardInterrupt:
        print("\n\nStopped viewing")
    
    print("\n💡 Use this to help create labels.csv")
    print("   Run: python scripts/ml_extraction/create_labels_helper.py (option 1)")


if __name__ == "__main__":
    print("="*80)
    print("📝 LABELS HELPER - Create labels.csv for Training")
    print("="*80)
    
    print("\nOptions:")
    print("1. Create labels.csv template (fill in manually)")
    print("2. View samples (to help with labeling)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        create_labels_template()
    elif choice == "2":
        view_samples_for_labeling()
    else:
        print("Invalid choice. Creating template...")
        create_labels_template()



