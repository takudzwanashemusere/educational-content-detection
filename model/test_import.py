print("Testing all imports and functions...\n")

try:
    from preprocessing import extract_frames, extract_audio_features
    print("✓ preprocessing — extract_frames imported")
    print("✓ preprocessing — extract_audio_features imported")
except Exception as e:
    print(f"✗ preprocessing — ERROR: {e}")

try:
    from model import build_video_classifier, get_compiled_model
    print("✓ model — build_video_classifier imported")
    print("✓ model — get_compiled_model imported")
except Exception as e:
    print(f"✗ model — ERROR: {e}")

try:
    from train import augment_frames, load_training_data, load_validation_data
    print("✓ train — augment_frames imported")
    print("✓ train — load_training_data imported")
    print("✓ train — load_validation_data imported")
except Exception as e:
    print(f"✗ train — ERROR: {e}")

try:
    import numpy as np
    dummy = np.zeros((20, 112, 112, 3), dtype=np.float32)
    result = augment_frames(dummy)
    assert len(result) == 4, "Expected 4 augmented versions"
    print("✓ augment_frames — returns 4 versions correctly")
except Exception as e:
    print(f"✗ augment_frames — ERROR: {e}")

try:
    import subprocess
    r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if r.returncode == 0:
        print("✓ ffmpeg — installed and working")
    else:
        print("✗ ffmpeg — NOT found. Install ffmpeg and add it to PATH")
except FileNotFoundError:
    print("✗ ffmpeg — NOT found. Install ffmpeg and add it to PATH")

print("\nAll checks complete.")
print("If all lines show ✓ you are ready to run: python train.py")
