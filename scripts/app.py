import os
import uuid
import base64
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store scale factors per session
scale_factors_store = {}

# Initialize the furniture detection processor
from utils.processing import FurnitureDetectionProcessor
processor = FurnitureDetectionProcessor(os.path.dirname(os.path.abspath(__file__)), debug=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image(image_path):
    """Resize image to exactly 800x600, maintaining aspect ratio and filling the frame completely"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Target dimensions
    target_width, target_height = 800, 600
    
    # Get original dimensions
    img_height, img_width = img.shape[:2]
    
    # Calculate aspect ratios
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height
    
    # Resize strategy to fill the frame
    if img_aspect > target_aspect:
        # Image is wider than target: fit height, crop width
        new_height = target_height
        new_width = int(new_height * img_aspect)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Crop center to target width
        start_x = (new_width - target_width) // 2
        resized = resized[:, start_x:start_x+target_width]
    else:
        # Image is taller than target: fit width, crop height
        new_width = target_width
        new_height = int(new_width / img_aspect)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Crop center to target height
        start_y = (new_height - target_height) // 2
        resized = resized[start_y:start_y+target_height, :]
    
    # Ensure exactly 800x600
    final_resized = cv2.resize(resized, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Create unique identifier for the resized image
    unique_id = str(uuid.uuid4())
    resized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"resized_{unique_id}.jpg")
    cv2.imwrite(resized_path, final_resized)
    
    return resized_path

def convert_length(feet):
    """Convert feet to centimeters"""
    return feet * 30.48  # Convert feet to cm

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and resize an image for point selection"""
    print("Upload endpoint called")
    
    if 'image' not in request.files:
        print("No image found in request.files")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    print(f"File received: {file.filename}")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No image selected'}), 400
        
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        
        try:
            # Resize the image to 800x600
            print("Resizing image")
            resized_path = resize_image(filepath)
            print(f"Image resized: {resized_path}")
            
            # Generate a session ID for this image
            session_id = str(uuid.uuid4())
            print(f"Session ID generated: {session_id}")
            
            # Store the resized path with the session ID
            scale_factors_store[session_id] = {
                'resized_path': resized_path,
                'scale_x': None,
                'scale_y': None
            }
            
            # Add the image as base64 for the frontend
            with open(resized_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            print("Sending response to client")
            # Return the session ID and image
            return jsonify({
                'session_id': session_id,
                'image': image_base64,
                'message': 'Please select points on the image to calculate scale factors'
            })
        
        except Exception as e:
            import traceback
            print(f"Error in image upload: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    print(f"Invalid file format: {file.filename}")
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/calculate-scale', methods=['POST'])
def calculate_scale():
    """Calculate scale factors based on selected points"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    session_id = data.get('session_id')
    points = data.get('points')
    x_length_feet = data.get('x_length_feet')
    y_length_feet = data.get('y_length_feet')
    
    if not session_id or session_id not in scale_factors_store:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    if not points or len(points) != 4:
        return jsonify({'error': 'Please select exactly 4 points (2 for X, 2 for Y)'}), 400
    
    if not x_length_feet or not y_length_feet:
        return jsonify({'error': 'Please provide lengths for both X and Y directions'}), 400
    
    try:
        # Calculate Manhattan distance for X direction
        dx = abs(points[0]['x'] - points[1]['x'])
        
        # Calculate Manhattan distance for Y direction
        dy = abs(points[2]['y'] - points[3]['y'])
        
        # Convert feet to centimeters
        x_length_cm = convert_length(float(x_length_feet))
        y_length_cm = convert_length(float(y_length_feet))
        
        # Calculate scale factors (cm/pixel)
        scale_x = x_length_cm / dx if dx != 0 else 0
        scale_y = y_length_cm / dy if dy != 0 else 0
        
        # Store scale factors
        scale_factors_store[session_id]['scale_x'] = scale_x
        scale_factors_store[session_id]['scale_y'] = scale_y
        
        return jsonify({
            'success': True,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'message': 'Scale factors calculated successfully'
        })
    
    except Exception as e:
        import traceback
        print(f"Error calculating scale factors: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# In your app.py file, update the /api/detect endpoint
@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for furniture detection using calculated scale factors"""
    print("=== /api/detect endpoint called ===")
    
    try:
        data = request.json
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        print(f"Session ID: {session_id}")
        
        if not session_id or session_id not in scale_factors_store:
            print(f"Invalid session ID. Available sessions: {list(scale_factors_store.keys())}")
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session_data = scale_factors_store[session_id]
        print(f"Session data: {session_data}")
        
        if not session_data['scale_x'] or not session_data['scale_y']:
            print("Scale factors not calculated yet")
            return jsonify({'error': 'Scale factors not calculated yet'}), 400
        
        resized_path = session_data['resized_path']
        print(f"Resized image path: {resized_path}")
        
        if not os.path.exists(resized_path):
            print(f"Image file not found: {resized_path}")
            return jsonify({'error': 'Image file not found'}), 404
        
        # Update the processor scale factors
        print(f"Setting scale factors: {session_data['scale_x']}, {session_data['scale_y']}")
        processor.scale_factors = {
            'scale_x': session_data['scale_x'],
            'scale_y': session_data['scale_y']
        }
        
        # Process the image using our comprehensive processor
        print("Starting image processing")
        results = processor.process_image(resized_path)
        print("Image processing completed")
        
        # Format the results for the frontend
        print("Formatting detection results")
        detections = format_detections(results)
        print(f"Formatted {len(detections)} detections")
        
        # Add the image as base64 for the frontend
        with open(resized_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Create final response
        response = {
            'image': image_base64,
            'detections': detections,
            'furniture_count': len(results['furniture']) - 1 if 'furniture' in results else 0,
            'wall_door_count': len(results['wall_door']) - 1 if 'wall_door' in results else 0,
            'scale_factors': {
                'scale_x': session_data['scale_x'],
                'scale_y': session_data['scale_y']
            }
        }
        
        print("Sending response to client")
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(f"ERROR in /api/detect: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

def format_detections(results):
    """Format the detection results for the frontend"""
    detections = []
    
    # Add furniture detections
    if 'furniture' in results:
        for key, item in results['furniture'].items():
            if key != "A1" and "name" in item:
                detections.append({
                    "class_name": item["name"],
                    "type": "furniture",
                    "confidence": item["confidence"],
                    "box": {
                        "x1": item["x1"],
                        "y1": item["y1"],
                        "x2": item["x2"],
                        "y2": item["y2"],
                        "width": item["width"],
                        "height": item["height"]
                    },
                    "orientation": item.get("orientation", "Unknown"),
                    "facing": item.get("facing", "Unknown")
                })
    
    # Add wall/door detections
    if 'wall_door' in results:
        for key, item in results['wall_door'].items():
            if key != "A1" and "name" in item:
                # Only add items that weren't already added from furniture
                existing = False
                for d in detections:
                    if (d["class_name"] == item["name"] and
                        abs(d["box"]["x1"] - item["x1"]) < 5 and
                        abs(d["box"]["y1"] - item["y1"]) < 5):
                        existing = True
                        break
                
                if not existing:
                    detections.append({
                        "class_name": item["name"],
                        "type": "wall_door",
                        "confidence": item["confidence"],
                        "box": {
                            "x1": item["x1"],
                            "y1": item["y1"],
                            "x2": item["x2"],
                            "y2": item["y2"],
                            "width": item["width"],
                            "height": item["height"]
                        },
                        "orientation": item.get("orientation", "Unknown"),
                        "facing": item.get("facing", "Unknown")
                    })
    
    return detections

@app.route('/output/<path:filename>')
def output_file(filename):
    """Serve files from the output directory"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)