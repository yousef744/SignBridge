#!/usr/bin/env python3
"""
SignBridge Pro - ML Backend
Flask server with trained ASL model
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
import base64
import io
import numpy as np
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from PIL import Image
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global model variable
model = None
label_mapping = None

# ASL Labels (fallback)
ASL_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'space'
}

def load_model():
    """Load trained model"""
    global model, label_mapping

    model_paths = [
        'models/asl_model.keras',
        'models/asl_model.h5',
        'models/best_model.h5'
    ]

    # Try to load model
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Model loaded from: {path}")
                break
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")

    # Load label mapping
    if os.path.exists('models/label_mapping.json'):
        with open('models/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            # Convert string keys to int
            label_mapping = {int(k): v for k, v in label_mapping.items()}
    else:
        label_mapping = ASL_LABELS

    if model is None:
        print("⚠️  No trained model found! Using mock predictions.")
        print("   Run: python train_model.py")

    return model is not None

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to grayscale
        image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize
        img_array = img_array.astype('float32') / 255.0

        # Reshape for model (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

def predict_sign(image_data):
    """Predict ASL sign from image"""

    if model is None:
        # Fallback to mock prediction
        import random
        classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space']
        pred_idx = random.randint(0, 26)
        confidence = random.uniform(0.75, 0.99)

        # Generate top-3 predictions
        top3 = []
        available = list(range(27))
        available.remove(pred_idx)
        for _ in range(3):
            idx = random.choice(available)
            available.remove(idx)
            top3.append({
                'letter': label_mapping.get(idx, ASL_LABELS.get(idx, '?')),
                'confidence': random.uniform(0.1, 0.4)
            })

        return {
            'success': True,
            'prediction': label_mapping.get(pred_idx, ASL_LABELS.get(pred_idx, '?')),
            'confidence': confidence,
            'top3': top3,
            'model_loaded': False
        }

    # Preprocess image
    processed = preprocess_image(image_data)

    if processed is None:
        return {
            'success': False,
            'error': 'Failed to preprocess image'
        }

    # Predict
    predictions = model.predict(processed, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])

    # Get top-3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3 = []
    for idx in top3_indices:
        top3.append({
            'letter': label_mapping.get(int(idx), ASL_LABELS.get(int(idx), '?')),
            'confidence': float(predictions[0][idx])
        })

    return {
        'success': True,
        'prediction': label_mapping.get(int(pred_idx), ASL_LABELS.get(int(pred_idx), '?')),
        'confidence': confidence,
        'top3': top3,
        'model_loaded': True
    }

# ===== ROUTES =====

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/status')
def status():
    """API status"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'model_type': 'CNN' if model else 'Mock',
        'classes': list(label_mapping.values()) if label_mapping else list(ASL_LABELS.values()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict ASL sign from image"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        image_data = data['image']
        result = predict_sign(image_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple frames"""
    try:
        data = request.get_json()

        if not data or 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400

        images = data['images']
        results = []

        for img_data in images:
            result = predict_sign(img_data)
            results.append(result)

        # Get most common prediction
        predictions = [r['prediction'] for r in results if r.get('success')]
        if predictions:
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0][0]
            avg_confidence = np.mean([r['confidence'] for r in results if r.get('success')])
        else:
            most_common = '-'
            avg_confidence = 0

        return jsonify({
            'success': True,
            'prediction': most_common,
            'confidence': float(avg_confidence),
            'frames_processed': len(images),
            'all_results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save', methods=['POST'])
def save_translation():
    """Save translation to file"""
    try:
        data = request.get_json()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs('translations', exist_ok=True)
        filename = f"translations/translation_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return jsonify({
            'success': True,
            'filename': filename
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 SignBridge Pro - ML Server")
    print("=" * 60)

    # Load model
    model_loaded = load_model()

    print(f"\n📊 Model Status: {'✅ Loaded' if model_loaded else '⚠️  Mock Mode'}")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
