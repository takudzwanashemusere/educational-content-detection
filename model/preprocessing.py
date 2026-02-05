import cv2
import numpy as np

def extract_frames(video_path, max_frames=10, frame_size=(112, 112)): #changed
    """
    Extract frames from video for model input
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"[WARN] Could not read video: {video_path}")
        cap.release()
        return None

    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    # If we got fewer than max_frames, pad with last frame
    if len(frames) == 0:
        print(f"[WARN] No frames extracted from: {video_path}")
        return None

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)
