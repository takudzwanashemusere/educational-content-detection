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

# Three-zone classification thresholds:
#   score >= ACCEPT_THRESHOLD  → approved (clearly educational)
#   score >= REVIEW_THRESHOLD  → under_review (AI uncertain, needs human check)
#   score <  REVIEW_THRESHOLD  → rejected (clearly not educational)
ACCEPT_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.50


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
            # Older gdown versions don't support fuzzy= — fall back without it
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



@app.route('/')
def home():
    model_status = 'ready' if model is not None else 'loading'
    return jsonify({
        'status': 'running',
        'model':  model_status,
        'message': 'Reelscholar Video Validation API',
        'endpoints': {
            'validate': '/api/validate-video  [POST]',
            'upload':   '/api/upload          [POST]'
        }
    })


def process_job(job_id, temp_path, filename, mode):
    """Runs in a background thread — processes video and stores result in jobs dict."""
    try:
       

        if model is None:
            jobs[job_id] = {'status': 'failed', 'error': 'Model not loaded.'}
            return

        frames = extract_frames(temp_path)
        if frames is None:
            jobs[job_id] = {'status': 'failed', 'error': 'Could not read video frames.'}
            return

        audio = extract_audio_features(temp_path)
        if audio is None:
            jobs[job_id] = {'status': 'failed', 'error': 'Could not extract audio.'}
            return

        frames_batch = np.expand_dims(frames, axis=0)
        audio_batch  = np.expand_dims(audio,  axis=0)

        prediction = model.predict(
            {'video_input': frames_batch, 'audio_input': audio_batch},
            verbose=0
        )[0][0]

        raw_score = float(prediction)

        if raw_score >= ACCEPT_THRESHOLD:
            verdict = 'approved'
        elif raw_score >= REVIEW_THRESHOLD:
            verdict = 'under_review'
        else:
            verdict = 'rejected'

        if mode == 'upload':
            if verdict == 'approved':
                result = {'success': True,  'status': 'approved',
                          'message': 'Your video has been uploaded successfully.'}
            elif verdict == 'under_review':
                result = {'success': True,  'status': 'under_review',
                          'message': ('Your video has been received and is currently '
                                      'under review. It will go live once our team '
                                      'has verified it.')}
            else:
                result = {'success': False, 'status': 'rejected',
                          'message': ('Your video could not be uploaded. '
                                      'Only educational content is allowed.')}
        else:
            if verdict == 'approved':
                confidence = raw_score
                message    = 'Video accepted — Educational content detected'
            elif verdict == 'under_review':
                confidence = raw_score
                message    = 'Video flagged for human review — AI confidence too low'
            else:
                confidence = 1.0 - raw_score
                message    = 'Video rejected — Not educational content'
            result = {'success': True, 'status': verdict, 'message': message, 'confidence': round(confidence, 4)}

        jobs[job_id] = {'status': 'done', 'result': result}

    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'error': str(e)}
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def submit_job(mode):
    """Shared entry point for both routes — saves file, starts background thread."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided.'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video selected.'}), 400

    ext = os.path.splitext(video_file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False,
                        'error': f'Unsupported file type "{ext}".'}), 400

    # Save file before background thread starts (request context ends after return)
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


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found.'}), 404
    if job['status'] == 'processing':
        return jsonify({'success': True, 'status': 'processing',
                        'message': 'Still processing, check back shortly.'}), 202
    if job['status'] == 'failed':
        return jsonify({'success': False, 'error': job.get('error')}), 500
    # done — return result and clean up
    result = job['result']
    del jobs[job_id]
    return jsonify(result), 200


@app.route('/api/validate-video', methods=['POST'])
def validate_video():
    return submit_job('validate')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    return submit_job('upload')


# Load model in a background thread so gunicorn workers are ready
# immediately (Render health checks won't block waiting for TF to load)


print("=" * 60)
print("  ReelScholar — Educational Video Validation API")
print("=" * 60)
threading.Thread(target=load_model, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server at http://0.0.0.0:{port}")
    print("Press CTRL+C to stop.")
    app.run(debug=False, port=port, host='0.0.0.0')