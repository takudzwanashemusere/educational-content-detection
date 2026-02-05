from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import tempfile
import os
from preprocessing import extract_frames


# Create Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = '../dataset/model/best_model.h5'

model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


@app.route('/')
def home():
    """Home endpoint to check if API is running"""
    return jsonify({
        'status': 'running',
        'message': 'CUT LearningHub Video Validation API',
        'endpoints': {
            'validate': '/api/validate-video [POST]',
            'upload': '/api/upload [POST]'
        }
    })


@app.route('/api/validate-video', methods=['POST'])
def validate_video():
    """
    API endpoint to validate if uploaded video is educational
    
    Expected: POST request with video file
    Returns: JSON with validation result
    """
    
    # Check if video file is in request
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file provided. Please upload a video.'
        }), 400
    
    video_file = request.files['video']
    
    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No video file selected'
        }), 400
    
  
    temp_path = None
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            temp_path = tmp.name
        
        print(f"Processing video: {video_file.filename}")
        
       
        frames = extract_frames(temp_path)
        frames = np.expand_dims(frames, axis=0) #chaned take a loookk
        
        if frames is None:
            return jsonify({
                'success': False,
                'error': 'Could not process video. Please check video format.'
            }), 400
        
        # Make prediction
        prediction = model.predict(frames, verbose=0)[0][0]
        is_educational = prediction > 0.5
        confidence = float(prediction if is_educational else 1 - prediction)
        
        # Prepare response
        result = {
            'success': True,
            'is_educational': bool(is_educational),
            'confidence': round(confidence * 100, 2),
            'message': 'Video accepted - Educational content detected' if is_educational 
                      else 'Video rejected - Not educational content'
        }
        
        print(f"Result: {result['message']} (Confidence: {result['confidence']}%)")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Validation error: {str(e)}'
        }), 500
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """
    Complete upload workflow with validation
    Only accepts educational videos
    """
    
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file provided'
        }), 400
    
    video_file = request.files['video']
    
    # First, validate the video
    validation_result = validate_video()
    validation_data = validation_result.get_json()
    
    if not validation_data.get('is_educational'):
        return jsonify({
            'success': False,
            'message': 'Upload rejected: Only educational content is allowed on CUT LearningHub',
            'confidence': validation_data.get('confidence')
        }), 403
    
    # If video is educational, proceed with upload
    # TODO: Add your actual upload logic here (save to database, cloud storage, etc.)
    
    return jsonify({
        'success': True,
        'message': 'Video uploaded successfully!',
        'video_id': 'temp_id_123',  # Replace with actual video ID
        'confidence': validation_data.get('confidence')
    })


if __name__ == '__main__':
    print("="*60)
    print("CUT LearningHub - Educational Video Validation API")
    print("="*60)
    
    # Load model before starting server
    if load_model():
        print("\nStarting Flask server...")
        print("API will be available at: http://127.0.0.1:5000")
        print("\nEndpoints:")
        print("  - GET  /                    (Check API status)")
        print("  - POST /api/validate-video  (Validate video)")
        print("  - POST /api/upload          (Upload video)")
        print("\nPress CTRL+C to stop the server")
        print("="*60)
        
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("\n Failed to start: Could not load model")
        print("Please train your model first by running: python train.py")
