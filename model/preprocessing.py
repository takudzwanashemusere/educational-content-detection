import cv2
import numpy as np
import librosa
import subprocess
import tempfile
import os

# Constants — must match across ALL files. If you change any value, delete
# best_model.h5 and retrain from scratch.
MAX_FRAMES       = 20
FRAME_SIZE       = (112, 112)
AUDIO_SR         = 22050
AUDIO_DURATION   = 10
AUDIO_N_MELS     = 64
AUDIO_HOP_LENGTH = 512
AUDIO_TIME_STEPS = 431   # = 1 + floor((10 * 22050) / 512)


def extract_frames(video_path, max_frames=MAX_FRAMES, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[WARN] Video has 0 frames: {video_path}")
        cap.release()
        return None
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frame = frame / 255.0
            frames.append(frame)
    cap.release()
    if len(frames) == 0:
        print(f"[WARN] No frames extracted from: {video_path}")
        return None
    while len(frames) < max_frames:
        frames.append(frames[-1])
    return np.array(frames, dtype=np.float32)


def extract_audio_features(
    video_path,
    duration=AUDIO_DURATION,
    sr=AUDIO_SR,
    n_mels=AUDIO_N_MELS,
    hop_length=AUDIO_HOP_LENGTH,
    fixed_time_steps=AUDIO_TIME_STEPS
):
    """
    Extract a mel spectrogram from the audio track of a video.
    Educational speech has very different audio patterns from music or
    crowd noise — this gives the model a second signal to learn from.
    Returns shape (n_mels, fixed_time_steps, 1) float32, or None on failure.
    """
    tmp_audio_path = None
    try:
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp_audio_path = tmp_audio.name
        tmp_audio.close()

        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_path,
             '-vn', '-acodec', 'pcm_s16le',
             '-ar', str(sr), '-ac', '1', tmp_audio_path],
            capture_output=True, timeout=60
        )
        if result.returncode != 0:
            print(f"[WARN] ffmpeg failed on: {video_path}")
            print(f"       {result.stderr.decode('utf-8', errors='ignore')[:300]}")
            return None

        audio, _ = librosa.load(tmp_audio_path, sr=sr, mono=True)
        if len(audio) == 0:
            print(f"[WARN] Empty audio in: {video_path}")
            return None

        # Clip from the MIDDLE of the video — avoids misleading intros/outros
        target_samples = duration * sr
        if len(audio) >= target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start: start + target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        mel_db   = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Fix width to exactly fixed_time_steps
        current = mel_norm.shape[1]
        if current < fixed_time_steps:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, fixed_time_steps - current)))
        elif current > fixed_time_steps:
            mel_norm = mel_norm[:, :fixed_time_steps]

        mel_norm = np.expand_dims(mel_norm, axis=-1)
        return mel_norm.astype(np.float32)

    except subprocess.TimeoutExpired:
        print(f"[WARN] ffmpeg timed out on: {video_path}")
        return None
    except Exception as e:
        print(f"[WARN] Audio extraction failed for {video_path}: {e}")
        return None
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)