"""
Flask Web Application for Text Extraction
Backend API for the front-end interface
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from pathlib import Path
import base64
import io
from PIL import Image
import os
from datetime import datetime

from main_pipeline import TextExtractionPipeline
from tesseract_ocr import TesseractOCR

app = Flask(__name__)
CORS(app)

# Initialize pipelines
print("Initializing text extraction pipelines...")
# Deep learning pipeline for complex images
deep_learning_pipeline = TextExtractionPipeline(filter_method='combined')
# Tesseract OCR for simpler images
tesseract_ocr = TesseractOCR(lang='eng+fas')
print("Pipelines ready!")

# Create uploads and outputs directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and encode image for preview
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Convert to base64 for display
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'message': 'Image uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_image():
    """Process image with selected filter and extract text"""
    try:
        data = request.json
        filename = data.get('filename')
        image_type = data.get('image_type', 'complex')  # 'simple' or 'complex'
        filter_method = data.get('filter', 'combined')
        confidence_threshold = float(data.get('confidence', 0.3))
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Route to appropriate OCR based on image type
        if image_type == 'simple':
            # Use Tesseract OCR with simpler preprocessing
            print(f"Processing as SIMPLE image with Tesseract OCR")
            
            # Apply simple filter (optional, lighter preprocessing)
            if filter_method != 'none':
                filtered_image = deep_learning_pipeline.filter_2d.apply_filter(image, method=filter_method)
            else:
                filtered_image = image
            
            # Extract text with Tesseract
            results = tesseract_ocr.extract_text(
                filtered_image,
                preprocess=True,  # Use simpler preprocessing
                config=None
            )
            
            ocr_method = 'Tesseract OCR'
            
        else:
            # Use Deep Learning OCR (EasyOCR) for complex images
            print(f"Processing as COMPLEX image with Deep Learning OCR")
            
            # Apply filter
            filtered_image = deep_learning_pipeline.filter_2d.apply_filter(image, method=filter_method)
            
            # Extract text with deep learning
            results = deep_learning_pipeline.text_extractor.extract_text(
                filtered_image,
                preprocess=False,
                confidence_threshold=confidence_threshold
            )
            
            ocr_method = 'Deep Learning (EasyOCR)'
        
        # Save filtered image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filtered_filename = f"filtered_{timestamp}.jpg"
        filtered_path = os.path.join(app.config['OUTPUT_FOLDER'], filtered_filename)
        cv2.imwrite(filtered_path, filtered_image)
        
        # Convert filtered image to base64
        _, buffer = cv2.imencode('.jpg', filtered_image)
        filtered_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'success': True,
            'filtered_image': f'data:image/jpeg;base64,{filtered_base64}',
            'filter_method': filter_method,
            'image_type': image_type,
            'ocr_method': ocr_method,
            'text_results': {
                'combined_text': results.get('combined_text', ''),
                'english_text': results.get('english_text', ''),
                'persian_text': results.get('persian_text', ''),
                'num_detections': len(results.get('text', [])),
                'detections': []
            }
        }
        
        # Add individual detections
        if results.get('text'):
            for i, (text, conf, lang) in enumerate(zip(
                results['text'],
                results.get('confidences', []),
                results.get('languages', [])
            )):
                response['text_results']['detections'].append({
                    'id': i + 1,
                    'text': text,
                    'confidence': float(conf) if isinstance(conf, (int, float)) else 0,
                    'language': lang
                })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/filters', methods=['GET'])
def get_filters():
    """Get list of available filters"""
    filters = [
        {'value': 'combined', 'name': 'Combined (Recommended)'},
        {'value': 'clahe', 'name': 'CLAHE'},
        {'value': 'adaptive_threshold', 'name': 'Adaptive Threshold'},
        {'value': 'morphological', 'name': 'Morphological'},
        {'value': 'top_hat', 'name': 'Top Hat'},
        {'value': 'gaussian_adaptive', 'name': 'Gaussian Adaptive'}
    ]
    return jsonify({'filters': filters})


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Text Extraction Web Application")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

