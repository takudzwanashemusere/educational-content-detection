import os
import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from model import get_compiled_model
from preprocessing import extract_frames, extract_audio_features


def augment_frames(frames):
    """4 augmented versions of one frame sequence. Audio is NOT augmented."""
    return [
        frames,
        np.flip(frames, axis=2).copy(),
        np.clip(frames * 1.2, 0.0, 1.0),
        np.clip(frames * 0.8, 0.0, 1.0),
    ]


def load_videos_from_dir(directory, label, apply_augmentation=True):
    X_video, X_audio, y = [], [], []

    if not os.path.exists(directory):
        print(f"[WARN] Directory not found: {directory}")
        return X_video, X_audio, y

    for video_file in sorted(os.listdir(directory)):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(directory, video_file)
        frames = extract_frames(video_path)
        audio  = extract_audio_features(video_path)

        if frames is None:
            print(f"  [SKIP] Could not extract frames: {video_file}")
            continue
        if audio is None:
            print(f"  [SKIP] Could not extract audio: {video_file}")
            continue

        if apply_augmentation:
            for aug_frames in augment_frames(frames):
                X_video.append(aug_frames)
                X_audio.append(audio)   # same audio for all augmented versions
                y.append(label)
            print(f"  Loaded + augmented (x4): {video_file}")
        else:
            X_video.append(frames)
            X_audio.append(audio)
            y.append(label)
            print(f"  Loaded: {video_file}")

    return X_video, X_audio, y


def load_training_data(data_dir):
    print("Loading educational training videos...")
    Xv_e, Xa_e, y_e = load_videos_from_dir(
        os.path.join(data_dir, 'train', 'educational'), 1, True)

    print("\nLoading non-educational training videos...")
    Xv_n, Xa_n, y_n = load_videos_from_dir(
        os.path.join(data_dir, 'train', 'non_educational'), 0, True)

    return (np.array(Xv_e + Xv_n, dtype=np.float32),
            np.array(Xa_e + Xa_n, dtype=np.float32),
            np.array(y_e  + y_n,  dtype=np.int32))


def load_validation_data(data_dir):
    print("Loading educational validation videos...")
    Xv_e, Xa_e, y_e = load_videos_from_dir(
        os.path.join(data_dir, 'validation', 'educational'), 1, False)

    print("\nLoading non-educational validation videos...")
    Xv_n, Xa_n, y_n = load_videos_from_dir(
        os.path.join(data_dir, 'validation', 'non_educational'), 0, False)

    return (np.array(Xv_e + Xv_n, dtype=np.float32),
            np.array(Xa_e + Xa_n, dtype=np.float32),
            np.array(y_e  + y_n,  dtype=np.int32))


def train_model():
    print("=" * 60)
    print("  Educational Video Classifier — Audio + Video Training")
    print("=" * 60)

    print("\n[1/5] Loading training data...")
    X_train_video, X_train_audio, y_train = load_training_data('../dataset')
    print(f"  Training samples: {len(y_train)}")

    if len(y_train) == 0:
        print("\n[ERROR] No training data found!")
        print("  Add videos to ../dataset/train/educational/")
        print("  and      ../dataset/train/non_educational/")
        return

    print("\n[2/5] Loading validation data...")
    X_val_video, X_val_audio, y_val = load_validation_data('../dataset')
    print(f"  Validation samples: {len(y_val)}")

    print("\n[3/5] Computing class weights...")
    cw_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(cw_array))
    print(f"  Class weights: {class_weight_dict}")

    print("\n[4/5] Building model...")
    model = get_compiled_model()
    model.summary()

    os.makedirs('../dataset/model', exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            '../dataset/model/best_model.h5',
            monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1)
    ]

    print("\n[5/5] Training...")
    model.fit(
        x={'video_input': X_train_video, 'audio_input': X_train_audio},
        y=y_train,
        validation_data=(
            {'video_input': X_val_video, 'audio_input': X_val_audio}, y_val
        ) if len(y_val) > 0 else None,
        epochs=30,
        batch_size=4,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    model.save('../dataset/model/final_model.h5')
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Best model : ../dataset/model/best_model.h5")
    print("=" * 60)


if __name__ == '__main__':
    train_model()