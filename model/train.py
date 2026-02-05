import os
import numpy as np
from tensorflow import keras
from model import get_compiled_model
from preprocessing import extract_frames


def load_training_data(data_dir):
    """
    Load and preprocess training videos
    
    Args:
        data_dir: Path to training data directory
    
    Returns:
        X_train, y_train arrays
    """
    X_train = []
    y_train = []
    
    # Load educational videos (label = 1)
    educational_dir = os.path.join(data_dir, 'train', 'educational')
    if os.path.exists(educational_dir):
        print("Loading educational videos...")
        for video_file in os.listdir(educational_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(educational_dir, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    X_train.append(frames)
                    y_train.append(1)
                    print(f"  Loaded: {video_file}")
                else:
                    print(f"[SKIP] Could not process: {video_file}")
    
    # Load non-educational videos (label = 0)
    non_educational_dir = os.path.join(data_dir, 'train', 'non_educational')
    if os.path.exists(non_educational_dir):
        print("Loading non-educational videos...")
        for video_file in os.listdir(non_educational_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(non_educational_dir, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    X_train.append(frames)
                    y_train.append(0)
                    print(f"  Loaded: {video_file}")
                else:
                    print(f"[SKIP] Could not process: {video_file}")
    
    return np.array(X_train), np.array(y_train)


def load_validation_data(data_dir):
    """
    Load and preprocess validation videos
    
    Args:
        data_dir: Path to validation data directory
    
    Returns:
        X_val, y_val arrays
    """
    X_val = []
    y_val = []
    
    # Load educational videos
    educational_dir = os.path.join(data_dir, 'validation', 'educational')
    if os.path.exists(educational_dir):
        print("Loading validation educational videos...")
        for video_file in os.listdir(educational_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(educational_dir, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    X_val.append(frames)
                    y_val.append(1)
                    print(f"  Loaded: {video_file}")
                else:
                    print(f"[SKIP] Could not process validation video: {video_file}")
    
    # Load non-educational videos
    non_educational_dir = os.path.join(data_dir, 'validation', 'non_educational')
    if os.path.exists(non_educational_dir):
        print("Loading validation non-educational videos...")
        for video_file in os.listdir(non_educational_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(non_educational_dir, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    X_val.append(frames)
                    y_val.append(0)
                    print(f"  Loaded: {video_file}")
                else:
                    print(f"[SKIP] Could not process validation video: {video_file}")
    
    return np.array(X_val), np.array(y_val)


def train_model():
    """
    Main training function
    """
    print("="*50)
    print("Starting Educational Video Classifier Training")
    print("="*50)
    
    # Load data
    print("\n1. Loading training data...")
    X_train, y_train = load_training_data('../dataset')
    
    print("\n2. Loading validation data...")
    X_val, y_val = load_validation_data('../dataset')
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    if len(X_train) == 0:
        print("\nERROR: No training data found!")
        print("Please add video files to ../dataset/train/educational/ and ../dataset/train/non_educational/")
        return
    
    # Create model
    print("\n3. Building model...")
    model = get_compiled_model()
    model.summary()
    
    # Setup callbacks
    os.makedirs('../dataset/model', exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            '../dataset/model/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=20,
        batch_size=1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n5. Saving final model...")
    model.save('../dataset/model/final_model.h5')
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best model saved to: ../dataset/model/best_model.h5")
    print("="*50)
    
    return history


if __name__ == '__main__':
    train_model()
