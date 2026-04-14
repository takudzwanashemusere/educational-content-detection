from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import tempfile
import os
from preprocessing import extract_frames, extract_audio_features

app = Flask(__name__)

# MODEL_PATH: local file path to use (default: best_model.h5 next to api.py)
# MODEL_URL:  Google Drive shareable link — if set, model is downloaded on startup
MODEL_PATH = os.environ.get('MODEL_PATH', './best_model.h5')
MODEL_URL  = os.environ.get('MODEL_URL', '')
model = None

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def download_model():
    """Download model from Google Drive if MODEL_URL is set and file is missing."""
    if not MODEL_URL:
        return
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already present at {MODEL_PATH}")
        return
    try:
        import gdown
        print(f"Downloading model from Google Drive → {MODEL_PATH} ...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("✓ Download complete.")
    except Exception as e:
        print(f"✗ Model download failed: {e}")


def load_model():
    global model
    download_model()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def run_validation(video_file):
    """
    Shared validation helper called by both routes.
    Extracts video frames AND audio, runs both through the model.
    File stream is only read once — avoids Flask stream consumption bug.
    """
    temp_path = None
    try:
        # Validate file extension before any processing
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {'success': False,
                    'error': f'Unsupported file type "{ext}". '
                             f'Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}, 400

        # Guard against model not being loaded
        if model is None:
            return {'success': False,
                    'error': 'Model not loaded. Run train.py first.'}, 503

        # Preserve the original extension so ffmpeg identifies the container correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            video_file.save(tmp.name)
            temp_path = tmp.name

        print(f"Processing: {video_file.filename}")

        frames = extract_frames(temp_path)
        if frames is None:
            return {'success': False,
                    'error': 'Could not read video frames. Check video format.'}, 400

        audio = extract_audio_features(temp_path)
        if audio is None:
            return {'success': False,
                    'error': ('Could not extract audio. '
                              'Make sure ffmpeg is installed and the video has audio.')}, 400

        # Add batch dimension to both inputs
        frames_batch = np.expand_dims(frames, axis=0)  # (1, 20, 112, 112, 3)
        audio_batch  = np.expand_dims(audio,  axis=0)  # (1, 64, 431, 1)

        prediction = model.predict(
            {'video_input': frames_batch, 'audio_input': audio_batch},
            verbose=0
        )[0][0]

        # Threshold raised to 0.75 to reduce false positives.
        # Video must be 75%+ likely to be educational to be accepted.
        is_educational = bool(prediction > 0.75)
        confidence     = float(prediction if is_educational else 1.0 - prediction)

        result = {
            'success': True,
            'is_educational': is_educational,
            'confidence': round(confidence * 100, 2),
            'raw_score': round(float(prediction) * 100, 2),
            'message': ('Video accepted — Educational content detected'
                        if is_educational
                        else 'Video rejected — Not educational content')
        }
        print(f"Result: {result['message']} (Confidence: {result['confidence']}%)")
        return result, 200

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return {'success': False, 'error': f'Validation error: {str(e)}'}, 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Reelscholar Video Validation API',
        'endpoints': {
            'validate': '/api/validate-video  [POST]',
            'upload':   '/api/upload          [POST]'
        }
    })


@app.route('/api/validate-video', methods=['POST'])
def validate_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided.'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video selected.'}), 400
    result, status_code = run_validation(video_file)
    return jsonify(result), status_code


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided.'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video selected.'}), 400

    result, status_code = run_validation(video_file)

    if not result.get('success'):
        return jsonify(result), status_code

    if not result.get('is_educational'):
        return jsonify({
            'success': False,
            'message': 'Upload rejected: Only educational content is allowed.',
            'confidence': result.get('confidence'),
            'raw_score':  result.get('raw_score')
        }), 403

    return jsonify({
        'success': True,
        'message': 'Video uploaded successfully!',
        'video_id': 'temp_id_123',
        'confidence': result.get('confidence'),
        'raw_score':  result.get('raw_score')
    }), 200


# Load model at startup (works for both gunicorn and direct python run)
print("=" * 60)
print("  ReelScholar — Educational Video Validation API")
print("=" * 60)
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server at http://0.0.0.0:{port}")
    print("Press CTRL+C to stop.")
    app.run(debug=False, port=port, host='0.0.0.0')