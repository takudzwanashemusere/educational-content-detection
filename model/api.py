from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import tempfile
import os
import uuid
import threading
from preprocessing import extract_frames, extract_audio_features

app = Flask(__name__)
CORS(app)

# In-memory job store: { job_id: { status, result } }
jobs = {}

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
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
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


def _score_message(score):
    """Return a human-readable label based on educational score percentage."""
    if score >= 80:
        return 'Highly educational content.'
    elif score >= 60:
        return 'Moderately educational content.'
    elif score >= 40:
        return 'Slightly educational content.'
    elif score >= 10:
        return 'Mostly non-educational content.'
    else:
        return 'Not educational — may contain inappropriate or irrelevant content.'


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    model_status = 'ready' if model is not None else 'loading'
    return jsonify({
        'status': 'running',
        'model':  model_status,
        'message': 'Reelscholar Video Validation API',
        'endpoints': {
            'validate_async': '/api/validate-video       [POST] — returns job_id, poll for result',
            'validate_sync':  '/api/validate-video-sync  [POST] — waits and returns score directly',
            'upload_async':   '/api/upload               [POST] — returns job_id, poll for result',
            'upload_sync':    '/api/upload-sync          [POST] — waits and returns score directly',
            'result':         '/api/result/<job_id>      [GET]  — poll async job result'
        }
    })


# ── Async helpers ─────────────────────────────────────────────────────────────

def process_job(job_id, temp_path, filename, mode):
    """Runs in a background thread — processes video and stores result in jobs dict."""
    try:
        if model is None:
            jobs[job_id] = {
                'status': 'failed',
                'http_code': 500,
                'error': 'Model not loaded. Please try again shortly.'
            }
            return

        frames = extract_frames(temp_path)
        if frames is None:
            jobs[job_id] = {
                'status': 'failed',
                'http_code': 500,
                'error': 'Server could not read video frames.'
            }
            return

        audio = extract_audio_features(temp_path)
        if audio is None:
            jobs[job_id] = {
                'status': 'failed',
                'http_code': 500,
                'error': 'Server could not extract audio from video.'
            }
            return

        frames_batch = np.expand_dims(frames, axis=0)
        audio_batch  = np.expand_dims(audio,  axis=0)

        prediction = model.predict(
            {'video_input': frames_batch, 'audio_input': audio_batch},
            verbose=0
        )[0][0]

        educational_score = round(float(prediction) * 100, 1)

        jobs[job_id] = {
            'status': 'done',
            'http_code': 200,
            'result': {
                'success': True,
                'filename': filename,
                'educational_score': educational_score,
                'message': _score_message(educational_score)
            }
        }

    except Exception as e:
        jobs[job_id] = {
            'status': 'failed',
            'http_code': 500,
            'error': f'Unexpected server error: {str(e)}'
        }
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def submit_job(mode):
    """Shared entry point for async routes — saves file, starts background thread."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided.'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video selected.'}), 400

    ext = os.path.splitext(video_file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False,
                        'error': f'Unsupported file type "{ext}". Allowed: mp4, avi, mov, mkv'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        video_file.save(tmp.name)
        temp_path = tmp.name

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing'}
    threading.Thread(target=process_job,
                     args=(job_id, temp_path, video_file.filename, mode),
                     daemon=True).start()

    return jsonify({'success': True, 'job_id': job_id,
                    'poll': f'/api/result/{job_id}'}), 202


# ── Sync helper ───────────────────────────────────────────────────────────────

def process_sync():
    """Shared entry point for sync routes — processes video and returns result immediately."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided.'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video selected.'}), 400

    ext = os.path.splitext(video_file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False,
                        'error': f'Unsupported file type "{ext}". Allowed: mp4, avi, mov, mkv'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        video_file.save(tmp.name)
        temp_path = tmp.name

    try:
        if model is None:
            return jsonify({'success': False,
                            'error': 'Model not loaded. Please try again shortly.'}), 500

        frames = extract_frames(temp_path)
        if frames is None:
            return jsonify({'success': False,
                            'error': 'Server could not read video frames.'}), 500

        audio = extract_audio_features(temp_path)
        if audio is None:
            return jsonify({'success': False,
                            'error': 'Server could not extract audio from video.'}), 500

        frames_batch = np.expand_dims(frames, axis=0)
        audio_batch  = np.expand_dims(audio,  axis=0)

        prediction = model.predict(
            {'video_input': frames_batch, 'audio_input': audio_batch},
            verbose=0
        )[0][0]

        educational_score = round(float(prediction) * 100, 1)

        return jsonify({
            'success': True,
            'filename': video_file.filename,
            'educational_score': educational_score,
            'message': _score_message(educational_score)
        }), 200

    except Exception as e:
        return jsonify({'success': False,
                        'error': f'Unexpected server error: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ── Async routes ──────────────────────────────────────────────────────────────

@app.route('/api/validate-video', methods=['POST'])
def validate_video():
    return submit_job('validate')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    return submit_job('upload')


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found.'}), 404
    if job['status'] == 'processing':
        return jsonify({'success': True, 'status': 'processing',
                        'message': 'Still processing, check back shortly.'}), 202
    if job['status'] == 'failed':
        return jsonify({'success': False,
                        'error': job.get('error')}), job.get('http_code', 500)
    result = job['result']
    del jobs[job_id]
    return jsonify(result), 200


# ── Sync routes ───────────────────────────────────────────────────────────────

@app.route('/api/validate-video-sync', methods=['POST'])
def validate_video_sync():
    return process_sync()


@app.route('/api/upload-sync', methods=['POST'])
def upload_video_sync():
    return process_sync()


# ── Startup ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("  ReelScholar — Educational Video Validation API")
print("=" * 60)

threading.Thread(target=load_model, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server at http://0.0.0.0:{port}")
    print("Press CTRL+C to stop.")
    app.run(debug=False, port=port, host='0.0.0.0')