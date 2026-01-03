"""
Train ML model to detect '0' in head column cells
Uses calibration samples from accurate_head_detector.py
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_training_data():
    """Load calibration samples and create labels"""
    samples_dir = Path("data/calibration_samples")
    
    if not samples_dir.exists():
        print("❌ Calibration samples not found!")
        print("   Run: python scripts/accurate_head_detector.py")
        print("   Choose option 5 (Manual calibration tool)")
        return None, None
    
    # Load images
    image_paths = sorted(list(samples_dir.glob("*.png")))
    
    if not image_paths:
        print("❌ No PNG files found in calibration_samples")
        return None, None
    
    print(f"📸 Found {len(image_paths)} calibration samples")
    
    # Check for labels file
    labels_path = samples_dir / "labels.csv"
    
    if not labels_path.exists():
        print("\n⚠️  No labels.csv found!")
        print("\n📝 CREATE LABELS FILE:")
        print("="*60)
        print("1. Look at each PNG in data/calibration_samples/")
        print("2. Determine if it contains '0' (head) or not")
        print("3. Create labels.csv with format:")
        print("\n   filename,is_zero")
        print("   sample_000_img0_row0.png,1")
        print("   sample_001_img0_row5.png,0")
        print("   sample_002_img0_row10.png,1")
        print("   ...")
        print("\n   Where:")
        print("   - 1 = contains '0' (head of household)")
        print("   - 0 = does not contain '0' (family member or empty)")
        print("\n💡 TIP: You can use Excel to create this file easily")
        return None, None
    
    # Load labels
    try:
        df = pd.read_csv(labels_path)
        
        if 'filename' not in df.columns or 'is_zero' not in df.columns:
            print("❌ labels.csv must have 'filename' and 'is_zero' columns")
            return None, None
        
    except Exception as e:
        print(f"❌ Error reading labels.csv: {e}")
        return None, None
    
    images = []
    labels = []
    missing_files = []
    
    for _, row in df.iterrows():
        img_path = samples_dir / row['filename']
        if img_path.exists():
            img = cv2.imread(str(img_path), 0)  # Grayscale
            if img is not None:
                # Resize to 50x50
                img_resized = cv2.resize(img, (50, 50))
                images.append(img_resized)
                
                # Convert is_zero to int (handle string '1'/'0' or boolean)
                is_zero = row['is_zero']
                if isinstance(is_zero, str):
                    is_zero = 1 if is_zero.strip() in ['1', 'True', 'true', 'yes'] else 0
                else:
                    is_zero = int(is_zero)
                
                labels.append(is_zero)
        else:
            missing_files.append(row['filename'])
    
    if missing_files:
        print(f"⚠️  Warning: {len(missing_files)} files from labels.csv not found")
        print(f"   First few: {missing_files[:5]}")
    
    if len(images) == 0:
        print("❌ No valid images loaded!")
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize
    images = images.astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    
    print(f"✅ Loaded {len(images)} labeled samples")
    print(f"   '0' samples (head): {np.sum(labels)}")
    print(f"   'not 0' samples: {len(labels) - np.sum(labels)}")
    
    # Check class balance
    if np.sum(labels) < 3 or (len(labels) - np.sum(labels)) < 3:
        print("⚠️  Warning: Very imbalanced dataset!")
        print("   You need at least a few samples of each class")
    
    return images, labels


def create_model():
    """Create CNN model for '0' detection"""
    model = keras.Sequential([
        keras.layers.Input(shape=(50, 50, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')  # Binary: '0' or not
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def train_model():
    """Train the '0' detection model"""
    print("="*80)
    print("🎓 TRAINING '0' DETECTION MODEL")
    print("="*80)
    
    # Load data
    X, y = load_training_data()
    
    if X is None:
        return
    
    # Check minimum data requirement
    if len(X) < 10:
        print("❌ Not enough training data!")
        print(f"   You have {len(X)} samples, need at least 10")
        print("   Create more calibration samples or label more existing ones")
        return
    
    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} samples ({np.sum(y_train)} '0', {len(y_train)-np.sum(y_train)} not '0')")
    print(f"   Val: {len(X_val)} samples ({np.sum(y_val)} '0', {len(y_val)-np.sum(y_val)} not '0')")
    
    # Create model
    model = create_model()
    
    print("\n🤖 Model architecture:")
    model.summary()
    
    # Callbacks
    model_path = Path("models/zero_detector.h5")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("\n🚀 Training...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=min(16, len(X_train) // 2),  # Adjust batch size
        verbose=1,
        callbacks=callbacks
    )
    
        # Evaluate
    print("\n📊 Final Results:")
    print("="*60)
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    
    # Unpack: [loss, accuracy, precision, recall]
    train_loss, train_acc, train_prec, train_rec = train_metrics
    val_loss, val_acc, val_prec, val_rec = val_metrics
    
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}, Precision: {train_prec:.2%}, Recall: {train_rec:.2%}")
    print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Precision: {val_prec:.2%}, Recall: {val_rec:.2%}")
    
    # Save final model
    model.save(str(model_path))
    print(f"\n✅ Model saved: {model_path}")
    
    # Save training history
    history_path = Path("models/training_history.json")
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    import json
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✅ Training history saved: {history_path}")
    
    # Plot training curves
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plot_path = Path("models/training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Training curves saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}")
    
    print("\n🎉 Training complete!")
    print("\nNext steps:")
    print("1. Check the model accuracy")
    print("2. Test on a few images: python scripts/ml_extraction/ml_based_extraction.py")
    print("3. If accuracy is good, run full extraction")


if __name__ == "__main__":
    train_model()



