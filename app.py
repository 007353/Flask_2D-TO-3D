import os
import uuid
import base64
import json
import cv2
import sys
import glob
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Configure CORS
allowed_origins = [
    'http://localhost:3000',  # React development server
    'https://flask-2d-to-3d.onrender.com',  # Render deployment
    'my-app-henna-seven-69.vercel.app',
    'my-8dcf953x1-007353s-projects.vercel.app'
]

CORS(app, resources={
    r"/api/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESIZED_FOLDER'] = 'static/resized'  # New dedicated folder for resized images
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESIZED_FOLDER'], exist_ok=True)  # Create resized folder
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store scale factors per session
scale_factors_store = {}

# Initialize the furniture detection processor
from utils.processing import FurnitureDetectionProcessor
processor = FurnitureDetectionProcessor(os.path.dirname(os.path.abspath(__file__)), debug=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image(image_path):
    """Resize image to exactly 800x600 preserving the entire image and aspect ratio"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Target dimensions
    target_width, target_height = 800, 600
    
    # Get original dimensions
    img_height, img_width = img.shape[:2]
    
    # Check if image is already 800x600
    if img_width == target_width and img_height == target_height:
        # Create a copy in the resized folder for consistent processing
        unique_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['RESIZED_FOLDER'], f"resized_{unique_id}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Image is already {target_width}x{target_height}, creating a copy in resized folder")
        return output_path
    
    print(f"Resizing image from {img_width}x{img_height} to {target_width}x{target_height} using INTER_AREA")
    
    # Create a white background canvas of 800x600
    resized = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    # Calculate aspect ratios
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height
    
    # Resize strategy to fit the entire image (not fill)
    if img_aspect > target_aspect:
        # Image is wider than target: fit to width
        new_width = target_width
        new_height = int(new_width / img_aspect)
        
        # Use INTER_AREA for downsampling to prevent aliasing artifacts
        temp = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate vertical offset to center the image
        y_offset = (target_height - new_height) // 2
        
        # Place the resized image on the white canvas
        resized[y_offset:y_offset+new_height, 0:new_width] = temp
        
    else:
        # Image is taller than target: fit to height
        new_height = target_height
        new_width = int(new_height * img_aspect)
        
        # Use INTER_AREA for downsampling to prevent aliasing artifacts
        temp = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate horizontal offset to center the image
        x_offset = (target_width - new_width) // 2
        
        # Place the resized image on the white canvas
        resized[0:new_height, x_offset:x_offset+new_width] = temp
    
    # Create unique identifier for the resized image
    unique_id = str(uuid.uuid4())
    resized_path = os.path.join(app.config['RESIZED_FOLDER'], f"resized_{unique_id}.jpg")
    cv2.imwrite(resized_path, resized)
    print(f"Saved resized image to: {resized_path}")
    print(f"FULL ABSOLUTE PATH TO RESIZED IMAGE: {os.path.abspath(resized_path)}")
    print(f"DOES FILE EXIST? {os.path.exists(resized_path)}")
    print(f"DIRECTORY CONTENTS: {os.listdir(app.config['RESIZED_FOLDER'])}")
    
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
            # Resize the image to 800x600 and save to resized folder
            print("Resizing image")
            resized_path = resize_image(filepath)
            print(f"Image resized: {resized_path}")
            
            # Remove the original uploaded file to save space and avoid confusion
            if os.path.exists(filepath) and filepath != resized_path:
                os.remove(filepath)
                print(f"Removed original file: {filepath}")
            
            # Verify the image is exactly 800x600
            img = cv2.imread(resized_path)
            if img is None:
                raise ValueError(f"Could not read resized image: {resized_path}")
                
            img_height, img_width = img.shape[:2]
            if img_width != 800 or img_height != 600:
                print(f"WARNING: Resized image dimensions {img_width}x{img_height} are not 800x600, re-resizing...")
                resized_path = resize_image(resized_path)
            
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
                'message': 'Please select points on the image to calculate scale factors',
                'dimensions': {
                    'width': 800,
                    'height': 600
                }
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
        print(f"Using resized image path from session: {resized_path}")
        
        if not os.path.exists(resized_path):
            print(f"Image file not found: {resized_path}")
            return jsonify({'error': 'Image file not found'}), 404
        
        # Verify the image is 800x600 before processing
        img = cv2.imread(resized_path)
        if img is None:
            print(f"Could not read image: {resized_path}")
            return jsonify({'error': 'Could not read image'}), 500
            
        img_height, img_width = img.shape[:2]
        if img_width != 800 or img_height != 600:
            print(f"Image dimensions {img_width}x{img_height} are not 800x600, resizing...")
            # If somehow the image isn't 800x600, resize it again
            resized_path = resize_image(resized_path)
            # Update the path in the session data
            scale_factors_store[session_id]['resized_path'] = resized_path
        
        # Update the processor scale factors
        print(f"Setting scale factors: {session_data['scale_x']}, {session_data['scale_y']}")
        processor.scale_factors = {
            'scale_x': session_data['scale_x'],
            'scale_y': session_data['scale_y']
        }
        
        # Process the image using our comprehensive processor
        print(f"Starting image processing on resized image: {resized_path}")
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
            'session_id': session_id,
            'image': image_base64,
            'detections': detections,
            'furniture_count': len(results['furniture']) - 1 if 'furniture' in results else 0,
            'wall_door_count': len(results['wall_door']) - 1 if 'wall_door' in results else 0,
            
            'room_count': len(results['room']) - 1 if 'room' in results else 0,
            'scale_factors': {
                'scale_x': session_data['scale_x'],
                'scale_y': session_data['scale_y']
            },
            'dimensions': {
                'width': 800,
                'height': 600
            }
        }
        
        print("Sending response to client")
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(f"ERROR in /api/detect: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

# Add route to serve resized images directly
@app.route('/static/resized/<path:filename>')
def resized_file(filename):
    """Serve files from the resized directory"""
    print(f"Serving resized file: {filename}")
    print(f"Full path: {os.path.join(app.config['RESIZED_FOLDER'], filename)}")
    print(f"File exists: {os.path.exists(os.path.join(app.config['RESIZED_FOLDER'], filename))}")
    
    # Make sure to use the absolute path instead of relative
    return send_from_directory(os.path.abspath(app.config['RESIZED_FOLDER']), filename)

# Alternatively, add this additional route that serves from the direct file path
@app.route('/direct-resized/<path:filename>')
def direct_resized_file(filename):
    """Serve resized files directly using absolute path"""
    print(f"Serving file directly: {filename}")
    # Since we're serving directly from the configured folder
    return send_from_directory(os.path.abspath(app.config['RESIZED_FOLDER']), filename)

@app.route('/static-index')
def static_index():
    """Show contents of static folders for troubleshooting"""
    try:
        # List contents of both static folders
        upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
        resized_files = os.listdir(app.config['RESIZED_FOLDER'])
        
        # Get absolute paths
        upload_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
        resized_path = os.path.abspath(app.config['RESIZED_FOLDER'])
        
        # Format HTML response
        html = f"""
        <html>
            <head>
                <title>Static Files Index</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    ul {{ list-style-type: none; padding: 0; }}
                    li {{ margin: 5px 0; padding: 5px; background: #f5f5f5; }}
                    .path {{ color: #777; font-style: italic; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>Static Files Index</h1>
                
                <h2>Upload Folder</h2>
                <div class="path">Path: {upload_path}</div>
                <ul>
                    {''.join([f'<li>{file}</li>' for file in upload_files]) if upload_files else '<li>No files</li>'}
                </ul>
                
                <h2>Resized Folder</h2>
                <div class="path">Path: {resized_path}</div>
                <ul>
                    {''.join([f'<li><a href="/direct-resized/{file}" target="_blank">{file}</a></li>' for file in resized_files]) if resized_files else '<li>No files</li>'}
                </ul>
            </body>
        </html>
        """
        
        return html
    except Exception as e:
        return f"Error listing files: {str(e)}"


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
    
    # Add room detections
    if 'room' in results:
        for key, item in results['room'].items():
            if key != "A1" and "class" in item:
                # Check if this is a duplicate (already added)
                existing = False
                for d in detections:
                    if (d["class_name"] == item["class"] and
                        abs(d["box"].get("x1", 0) - item.get("x1", 0)) < 5 and
                        abs(d["box"].get("y1", 0) - item.get("y1", 0)) < 5):
                        existing = True
                        break
                
                if not existing:
                    detections.append({
                        "class_name": item["class"],
                        "type": "room",
                        "confidence": item.get("confidence", 1.0),
                        "box": {
                            "x1": item.get("x1", 0),
                            "y1": item.get("y1", 0),
                            "x2": item.get("x2", 0),
                            "y2": item.get("y2", 0),
                            "x3": item.get("x3", 0),
                            "y3": item.get("y3", 0),
                            "x4": item.get("x4", 0),
                            "y4": item.get("y4", 0),
                            "width": item.get("width", 0),
                            "height": item.get("height", 0)
                        }
                    })
    
    return detections

@app.route('/output/<path:filename>')
def output_file(filename):
    """Serve files from the output directory"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/session-data/<session_id>', methods=['GET'])
def get_session_data(session_id):
    """Get session data for the given session ID"""
    print(f"Fetching session data for {session_id}")
    
    if not session_id or session_id not in scale_factors_store:
        print(f"Session ID {session_id} not found in store")
        return jsonify({'error': 'Invalid session ID'}), 404
    
    session_data = scale_factors_store[session_id]
    print(f"Found session data: {session_data}")
    
    # Convert the session data to a JSON response
    response = {
        'resized_path': session_data.get('resized_path', ''),
        'scale_x': session_data.get('scale_x'),
        'scale_y': session_data.get('scale_y')
    }
    
    print(f"Sending session data: {response}")
    return jsonify(response)

@app.route('/get-latest-files', methods=['GET'])
def get_latest_files():
    """
    Get the paths to the latest furniture and wall/door output files.
    Returns JSON with the paths to the most recent output files.
    """
    try:
        # The base directory with the output files
        output_dir = app.config['OUTPUT_FOLDER']
        
        # Log directory check to console
        print(f"Checking for output files in: {os.path.abspath(output_dir)}")
        if not os.path.exists(output_dir):
            print(f"Output directory does not exist: {os.path.abspath(output_dir)}")
            return jsonify({
                "error": "Output directory not found",
                "checked_path": os.path.abspath(output_dir)
            }), 404
        
        # Find all furniture_output files
        furniture_pattern = os.path.join(output_dir, 'furniture_output_*.json')
        furniture_files = glob.glob(furniture_pattern)
        
        # Find all wall_door_output files
        wall_door_pattern = os.path.join(output_dir, 'wall_door_output_*.json')
        wall_door_files = glob.glob(wall_door_pattern)
        
        # Log what was found
        print(f"Found {len(furniture_files)} furniture files and {len(wall_door_files)} wall/door files")
        
        if not furniture_files or not wall_door_files:
            return jsonify({
                "error": "No output files found",
                "furniture_found": len(furniture_files),
                "wall_door_found": len(wall_door_files)
            }), 404
        
        # Sort by modification time (newest first)
        latest_furniture = max(furniture_files, key=os.path.getmtime)
        latest_wall_door = max(wall_door_files, key=os.path.getmtime)
        
        # Get modification times for debugging
        furniture_time = datetime.fromtimestamp(os.path.getmtime(latest_furniture))
        wall_door_time = datetime.fromtimestamp(os.path.getmtime(latest_wall_door))
        
        # Return the paths
        return jsonify({
            "furniture_output": latest_furniture,
            "wall_door_output": latest_wall_door,
            "furniture_modified": furniture_time.isoformat(),
            "wall_door_modified": wall_door_time.isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_latest_files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-file', methods=['GET'])
def get_file():
    """
    Get the content of a specific file.
    Requires 'path' query parameter.
    """
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({"error": "No path provided"}), 400
            
        # Ensure the path is valid and within the output directory
        abs_file_path = os.path.abspath(file_path)
        abs_output_dir = os.path.abspath(app.config['OUTPUT_FOLDER'])
        
        if not abs_file_path.startswith(abs_output_dir) and not file_path.startswith(app.config['OUTPUT_FOLDER']):
            return jsonify({
                "error": "Invalid path - must be in the output directory",
                "provided_path": file_path,
                "output_dir": app.config['OUTPUT_FOLDER']
            }), 403
            
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
            
        # Read and return file content
        with open(file_path, 'r') as f:
            content = json.loads(f.read())
            
        return jsonify(content)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug-paths', methods=['GET'])
def debug_paths():
    """Debug endpoint to check if output folders exist and are accessible"""
    try:
        # Check current working directory
        cwd = os.getcwd()
        
        # Check if output folder exists
        output_dir = app.config['OUTPUT_FOLDER']
        abs_output_dir = os.path.abspath(output_dir)
        output_exists = os.path.exists(output_dir)
        
        # Check if there are any files in the output directory
        files_in_output = []
        if output_exists:
            try:
                files_in_output = os.listdir(output_dir)
            except Exception as e:
                files_in_output = [f"Error listing directory: {str(e)}"]
        
        # Get all session directories in the output folder
        session_dirs = []
        if output_exists:
            try:
                session_dirs = [d for d in os.listdir(output_dir) 
                              if os.path.isdir(os.path.join(output_dir, d))]
            except Exception as e:
                session_dirs = [f"Error listing sessions: {str(e)}"]
        
        # Check for combined_detections.json files in session folders
        combined_files = []
        for session in session_dirs:
            session_path = os.path.join(output_dir, session)
            combined_path = os.path.join(session_path, 'combined_detections.json')
            if os.path.exists(combined_path):
                try:
                    # Count items in the file
                    with open(combined_path, 'r') as f:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else 'unknown'
                    
                    combined_files.append({
                        "session_id": session,
                        "path": combined_path,
                        "box_count": count,
                        "modified": os.path.getmtime(combined_path)
                    })
                except Exception as e:
                    combined_files.append({
                        "session_id": session,
                        "path": combined_path,
                        "error": str(e)
                    })
        
        # Check permissions on output directory
        permissions = {
            "read": os.access(output_dir, os.R_OK) if output_exists else False,
            "write": os.access(output_dir, os.W_OK) if output_exists else False,
            "execute": os.access(output_dir, os.X_OK) if output_exists else False
        }
        
        # Return all the debug info
        return jsonify({
            "current_working_directory": cwd,
            "output_directory": output_dir,
            "absolute_output_directory": abs_output_dir,
            "output_directory_exists": output_exists,
            "output_permissions": permissions,
            "files_in_output_dir": files_in_output,
            "session_directories": session_dirs,
            "combined_detection_files": combined_files,
            "flask_config": {
                "upload_folder": app.config['UPLOAD_FOLDER'],
                "resized_folder": app.config['RESIZED_FOLDER'],
                "output_folder": app.config['OUTPUT_FOLDER']
            }
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Add this endpoint to your Flask application for reprocessing with threshold
@app.route('/reprocess-with-threshold', methods=['POST'])
def reprocess_with_threshold():
    try:
        # Get the confidence threshold from the request
        data = request.json
        confidence_threshold = data.get('confidence_threshold', 0.01)
        session_id = data.get('session_id')  # Added to get session_id from request
        
        # Validate confidence threshold
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            return jsonify({"error": "Invalid confidence threshold"}), 400
        
        # Log the threshold for debugging
        print(f"Reprocessing with confidence threshold: {confidence_threshold}, session_id: {session_id}")
        
        # Get the latest uploaded image path
        latest_image = get_latest_uploaded_image()
        if not latest_image:
            return jsonify({"error": "No image to reprocess"}), 400
        
        print(f"Reprocessing image: {latest_image}")
        
        # Get scale factors from the session store if available
        scale_x = 1.0
        scale_y = 1.0
        session_data = None
        
        # Try to get scale factors from the session store if available
        if session_id and session_id in scale_factors_store:
            session_data = scale_factors_store[session_id]
            if 'scale_x' in session_data and 'scale_y' in session_data and session_data['scale_x'] and session_data['scale_y']:
                scale_x = session_data['scale_x']
                scale_y = session_data['scale_y']
                print(f"Using scale factors from session {session_id}: scale_x={scale_x}, scale_y={scale_y}")
        else:
            # If no specific session_id provided, try to find a session with scale factors
            for sess_id, sess_data in scale_factors_store.items():
                if 'scale_x' in sess_data and 'scale_y' in sess_data and sess_data['scale_x'] and sess_data['scale_y']:
                    scale_x = sess_data['scale_x']
                    scale_y = sess_data['scale_y']
                    session_id = sess_id  # Use this session ID
                    print(f"Using scale factors from session {sess_id}: scale_x={scale_x}, scale_y={scale_y}")
                    break
        
        # Set the scale factors in the processor
        processor.scale_factors = {
            "scale_x": scale_x,
            "scale_y": scale_y
        }
        
        # Check if there are separate manual box files for wall_door and furniture
        wall_door_manual = []
        furniture_manual = []
        manual_boxes_added = False
        
        if session_id:
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
            wall_door_manual_path = os.path.join(output_dir, 'wall_door_manual.json')
            furniture_manual_path = os.path.join(output_dir, 'furniture_manual.json')
            
            # Load wall_door manual boxes if available
            if os.path.exists(wall_door_manual_path):
                try:
                    with open(wall_door_manual_path, 'r') as f:
                        wall_door_manual = json.load(f)
                    print(f"Loaded {len(wall_door_manual)} wall_door manual boxes")
                    manual_boxes_added = True
                except Exception as e:
                    print(f"Error loading wall_door manual boxes: {str(e)}")
            
            # Load furniture manual boxes if available
            if os.path.exists(furniture_manual_path):
                try:
                    with open(furniture_manual_path, 'r') as f:
                        furniture_manual = json.load(f)
                    print(f"Loaded {len(furniture_manual)} furniture manual boxes")
                    manual_boxes_added = True
                except Exception as e:
                    print(f"Error loading furniture manual boxes: {str(e)}")
            
            # If separate files don't exist, try to load from combined manual boxes
            if not wall_door_manual and not furniture_manual:
                manual_boxes_path = os.path.join(output_dir, 'manual_boxes.json')
                if os.path.exists(manual_boxes_path):
                    try:
                        with open(manual_boxes_path, 'r') as f:
                            manual_boxes = json.load(f)
                        
                        # Separate boxes by type
                        for box in manual_boxes:
                            box_type = box.get('type', '').lower()
                            if box_type == 'wall_door':
                                wall_door_manual.append(box)
                            elif box_type == 'furniture':
                                furniture_manual.append(box)
                        
                        print(f"Loaded and separated {len(wall_door_manual)} wall_door and {len(furniture_manual)} furniture manual boxes")
                        manual_boxes_added = len(wall_door_manual) > 0 or len(furniture_manual) > 0
                    except Exception as e:
                        print(f"Error loading manual boxes: {str(e)}")
        
        # Pass the manual boxes to the processor
        processor.wall_door_manual_boxes = wall_door_manual
        processor.furniture_manual_boxes = furniture_manual
        
        # Process the image with the new confidence threshold and pass the session_id
        result = processor.process_image(
            latest_image, 
            confidence_threshold=confidence_threshold, 
            session_id=session_id,
            include_manual_boxes=manual_boxes_added
        )
        
        # Update the result with manual boxes status
        result["manual_boxes_added"] = manual_boxes_added
        result["manual_boxes_count"] = {
            "wall_door": len(wall_door_manual),
            "furniture": len(furniture_manual)
        }
        
        # Return success response with paths to the generated files
        return jsonify({
            "status": "success",
            "message": f"Image reprocessed with confidence threshold: {confidence_threshold}" + 
                      (", manual boxes included" if manual_boxes_added else ""),
            "furniture_file": "FURNITURE.json",
            "wall_file": "WALL.json",
            "manual_boxes_added": manual_boxes_added,
            "session_id": session_id,
            "detections_count": {
                "furniture": len(result["furniture_predictions"]["predictions"]) if "furniture_predictions" in result else 0,
                "wall_door": len(result["wall_door_predictions"]["predictions"]) if "wall_door_predictions" in result else 0,
                "manual_wall_door": len(wall_door_manual),
                "manual_furniture": len(furniture_manual)
            }
        })
        
    except Exception as e:
        print(f"Error reprocessing image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Helper function to get the most recently uploaded image
def get_latest_uploaded_image():
    upload_folder = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    resized_folder = os.path.join(app.root_path, app.config['RESIZED_FOLDER'])
    
    # First try to get the most recent resized image (preferred)
    resized_files = []
    
    for filename in os.listdir(resized_folder):
        file_path = os.path.join(resized_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            resized_files.append((file_path, os.path.getmtime(file_path)))
    
    # Then check the uploaded files as backup
    uploaded_files = []
    
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            uploaded_files.append((file_path, os.path.getmtime(file_path)))
    
    # Combine files and sort by modification time (newest first)
    all_files = resized_files + uploaded_files
    
    if not all_files:
        print("No image files found in either folder")
        return None
    
    all_files.sort(key=lambda x: x[1], reverse=True)
    latest_file_path = all_files[0][0]
    print(f"Found latest image: {latest_file_path}")
    
    return latest_file_path

@app.route('/get-latest-threshold', methods=['GET'])
def get_latest_threshold():
    try:
        # Check if combined_results.json exists which stores the last applied threshold
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'output')
        combined_results_path = os.path.join(output_dir, 'combined_results.json')
        
        applied_threshold = None
        
        # Check if we can find the applied threshold in combined_results.json
        if os.path.exists(combined_results_path):
            try:
                with open(combined_results_path, 'r') as f:
                    combined_data = json.load(f)
                    if 'confidence_threshold' in combined_data:
                        applied_threshold = combined_data['confidence_threshold']
                        print(f"Found applied threshold: {applied_threshold}")
            except Exception as e:
                print(f"Error reading combined_results.json: {str(e)}")
        
        # If threshold not found, check FURNITURE.json and WALL.json and make an estimate
        if applied_threshold is None:
            furniture_path = os.path.join(output_dir, 'FURNITURE.json')
            wall_path = os.path.join(output_dir, 'WALL.json')
            
            if os.path.exists(furniture_path) and os.path.exists(wall_path):
                try:
                    with open(furniture_path, 'r') as f:
                        furniture_data = json.load(f)
                    
                    with open(wall_path, 'r') as f:
                        wall_data = json.load(f)
                    
                    # Count number of items in each file to estimate threshold
                    item_count = sum(1 for key in furniture_data if key != "A1")
                    item_count += sum(1 for key in wall_data if key != "A1")
                    
                    # Make a rough estimate based on item count
                    if item_count < 3:
                        applied_threshold = 0.5  # Changed from 0.8 to 0.5
                    elif item_count < 6:
                        applied_threshold = 0.3  # Changed from 0.6 to 0.3
                    else:
                        applied_threshold = 0.01  # Changed from 0.5 to 0.01
                        
                    print(f"Estimated threshold based on item count ({item_count}): {applied_threshold}")
                except Exception as e:
                    print(f"Error estimating threshold: {str(e)}")
                    applied_threshold = 0.01  # Changed from 0.5 to 0.01
        
        # Default to 0.01 if we couldn't find or estimate
        if applied_threshold is None:
            applied_threshold = 0.01  # Changed from 0.5 to 0.01
            print("Using default threshold: 0.01")
        
        return jsonify({
            "status": "success",
            "applied_threshold": applied_threshold
        })
    except Exception as e:
        print(f"Error retrieving threshold: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add endpoint to apply threshold and upload to AWS
@app.route('/apply-threshold-and-upload', methods=['POST'])
def apply_threshold_and_upload():
    try:
        # Get the confidence threshold from the request
        data = request.json
        confidence_threshold = data.get('confidence_threshold', 0.01)
        
        # Validate confidence threshold
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            return jsonify({"error": "Invalid confidence threshold"}), 400
        
        # Log the threshold for debugging
        print(f"Applying threshold {confidence_threshold} and uploading")
        
        # First, reprocess the image with the new threshold
        # Get the latest image path
        latest_image = get_latest_uploaded_image()
        if not latest_image:
            return jsonify({"error": "No image to reprocess"}), 400
        
        print(f"Reprocessing image: {latest_image}")
        
        # Get scale factors
        scale_x = 1.0
        scale_y = 1.0
        
        # Try to get scale factors from the session store if available
        for session_id, session_data in scale_factors_store.items():
            if 'scale_x' in session_data and 'scale_y' in session_data and session_data['scale_x'] and session_data['scale_y']:
                scale_x = session_data['scale_x']
                scale_y = session_data['scale_y']
                print(f"Using scale factors from session {session_id}: scale_x={scale_x}, scale_y={scale_y}")
                break
        
        # Set the scale factors in the processor
        processor.scale_factors = {
            "scale_x": scale_x,
            "scale_y": scale_y
        }
        
        # Process the image with the new confidence threshold
        result = processor.process_image(latest_image, confidence_threshold=confidence_threshold)
        
        # Check if the processing was successful
        if not result:
            return jsonify({"error": "Failed to process image with new threshold"}), 500
        
        # Check if AWS upload paths were returned
        furniture_s3_path = None
        wall_s3_path = None
        
        
        if 'aws_paths' in result:
            furniture_s3_path = result['aws_paths'].get('furniture')
            wall_s3_path = result['aws_paths'].get('wall')
            
        # If not, try to upload the files directly
        if not furniture_s3_path or not wall_s3_path:
            # Get paths to the FURNITURE.json and WALL.json files
            base_dir = os.path.dirname(os.path.abspath(__file__))
            furniture_path = os.path.join(base_dir, 'output', 'FURNITURE.json')
            wall_path = os.path.join(base_dir, 'output', 'WALL.json')
            room_path = os.path.join(base_dir, 'output', 'ROOM.json')
            if os.path.exists(room_path):
              room_s3_path = processor.upload_to_aws_s3(room_path, "ROOM.json")
            else:
               room_s3_path = None
            
            
            if not os.path.exists(furniture_path) or not os.path.exists(wall_path):
                return jsonify({"error": "Processed files not found"}), 404
            
            # Upload the files to AWS S3
            furniture_s3_path = processor.upload_to_aws_s3(furniture_path, "FURNITURE.json")
            wall_s3_path = processor.upload_to_aws_s3(wall_path, "WALL.json")
        
        if not furniture_s3_path or not wall_s3_path:
            return jsonify({"error": "Failed to upload to AWS S3"}), 500
        
        # Save the combined results with threshold information for reference
        combined_results = {
            "confidence_threshold": confidence_threshold,
            "timestamp": datetime.now().isoformat(),
            "furniture_s3_path": furniture_s3_path,
            "wall_s3_path": wall_s3_path,
            "room_s3_path": room_s3_path,
            "scale_factors": {
                "scale_x": scale_x,
                "scale_y": scale_y
            }
        }
        
        # Make sure the output directory exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        combined_results_path = os.path.join(output_dir, 'combined_results.json')
        with open(combined_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        # Return success response with S3 paths
        return jsonify({
            "status": "success",
            "message": f"Files processed with threshold {confidence_threshold} and uploaded to AWS",
            "furniture_s3_path": furniture_s3_path,
            "wall_s3_path": wall_s3_path,
            "scale_factors": {
                "scale_x": scale_x,
                "scale_y": scale_y
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error applying threshold and uploading: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/save-manual-boxes', methods=['POST'])
def save_manual_boxes():
    try:
        data = request.get_json()
        print("Received data for manual boxes:", json.dumps(data, indent=2))  # Debug log
        
        session_id = data.get('session_id')
        boxes = data.get('boxes', [])

        if not session_id:
            print("Error: No session_id provided")
            return jsonify({'error': 'Missing session_id'}), 400

        if not boxes:
            print("Error: No boxes provided")
            return jsonify({'error': 'No boxes provided'}), 400

        print(f"Processing {len(boxes)} boxes for session {session_id}")
        
        # DEBUG: Print detailed box info
        for i, box in enumerate(boxes):
            print(f"Box {i} structure:", json.dumps(box, indent=2))
            # Check for keypoints
            if 'keypoints' in box and box['keypoints']:
                print(f"Box {i} has {len(box['keypoints'])} keypoints")
            # Check for TopX, TopY properties
            if 'TopX' in box and box['TopX'] is not None:
                print(f"Box {i} has TopX/TopY properties")
        
        # Create absolute output directory path 
        abs_output_folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
        output_dir = os.path.join(abs_output_folder, session_id)
        
        print(f"Target output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/verified output directory: {output_dir}")

        # Process each box to ensure keypoint information is properly formatted
        for box in boxes:
            # Ensure keypoints array exists
            if 'keypoints' not in box:
                box['keypoints'] = []
            
            # Determine if TopX, TopY, BottomX, BottomY are directly provided
            # If so, ensure they're also represented in the keypoints array
            if ('TopX' in box and box['TopX'] is not None and 
                'TopY' in box and box['TopY'] is not None):
                # Check if a 'top' keypoint already exists
                top_exists = any(kp.get('type') == 'top' for kp in box['keypoints'])
                if not top_exists:
                    box['keypoints'].append({
                        'type': 'top',
                        'x': box['TopX'],
                        'y': box['TopY']
                    })
            
            if ('BottomX' in box and box['BottomX'] is not None and 
                'BottomY' in box and box['BottomY'] is not None):
                # Check if a 'bottom' keypoint already exists
                bottom_exists = any(kp.get('type') == 'bottom' for kp in box['keypoints'])
                if not bottom_exists:
                    box['keypoints'].append({
                        'type': 'bottom',
                        'x': box['BottomX'],
                        'y': box['BottomY']
                    })
            
            # If keypoints exist, set TopX, TopY, BottomX, BottomY properties
            for kp in box['keypoints']:
                if kp.get('type') == 'top':
                    box['TopX'] = kp['x']
                    box['TopY'] = kp['y']
                elif kp.get('type') == 'bottom':
                    box['BottomX'] = kp['x']
                    box['BottomY'] = kp['y']

        # Save exact manual boxes as they were received (preserve all fields)
        manual_boxes_file = os.path.join(output_dir, 'manual_boxes.json')
        with open(manual_boxes_file, 'w') as f:
            json.dump(boxes, f, indent=2)
        print(f"Saved manual boxes to: {manual_boxes_file}")

        # Separate boxes by type
        wall_door_boxes = []
        furniture_boxes = []
        
        for box in boxes:
            box_type = box.get('type', '').lower()
            if box_type == 'wall_door':
                wall_door_boxes.append(box)
            elif box_type == 'furniture':
                furniture_boxes.append(box)
            else:
                print(f"Unknown box type: {box_type} for box: {box}")

        print(f"Separated boxes: {len(wall_door_boxes)} wall_door, {len(furniture_boxes)} furniture")

        # Update the combined detections file if it exists
        combined_file = os.path.join(output_dir, 'combined_detections.json')
        wall_door_file = os.path.join(output_dir, 'wall_door_manual.json')
        furniture_file = os.path.join(output_dir, 'furniture_manual.json')
        
        # Save separated manual boxes
        with open(wall_door_file, 'w') as f:
            json.dump(wall_door_boxes, f, indent=2)
        print(f"Saved wall_door manual boxes to: {wall_door_file}")
        
        with open(furniture_file, 'w') as f:
            json.dump(furniture_boxes, f, indent=2)
        print(f"Saved furniture manual boxes to: {furniture_file}")
        # Separate boxes by type
        wall_door_boxes = []
        furniture_boxes = []
        
        for box in boxes:
            box_type = box.get('type', '').lower()
            if box_type == 'wall_door':
                wall_door_boxes.append(box)
            elif box_type == 'furniture':
                furniture_boxes.append(box)
            else:
                print(f"Unknown box type: {box_type} for box: {box}")

        print(f"Separated boxes: {len(wall_door_boxes)} wall_door, {len(furniture_boxes)} furniture")

        # Update the combined detections file if it exists
        combined_file = os.path.join(output_dir, 'combined_detections.json')
        wall_door_file = os.path.join(output_dir, 'wall_door_manual.json')
        furniture_file = os.path.join(output_dir, 'furniture_manual.json')
        
        # Save separated manual boxes
        with open(wall_door_file, 'w') as f:
            json.dump(wall_door_boxes, f, indent=2)
        print(f"Saved wall_door manual boxes to: {wall_door_file}")
        
        with open(furniture_file, 'w') as f:
            json.dump(furniture_boxes, f, indent=2)
        print(f"Saved furniture manual boxes to: {furniture_file}")
        
        try:
            if os.path.exists(combined_file):
                print(f"Found existing combined detections file: {combined_file}")
                with open(combined_file, 'r') as f:
                    combined_detections = json.load(f)
                print(f"Loaded {len(combined_detections) if isinstance(combined_detections, list) else 'unknown'} existing detections")
            else:
                print("No existing combined detections file, creating new one")
                combined_detections = []
                
            # Add manual boxes to combined detections (avoid duplicates)
            if isinstance(combined_detections, list):
                # Create a simple signature for each existing box
                existing_sigs = set()
                for existing_box in combined_detections:
                    # Handle both nested and flat structures
                    if 'box' in existing_box and 'x1' in existing_box['box'] and 'y1' in existing_box['box']:
                        # Nested structure
                        x1 = existing_box['box']['x1']
                        y1 = existing_box['box']['y1']
                        class_name = existing_box.get('class_name', '')
                    elif 'x1' in existing_box and 'y1' in existing_box:
                        # Flat structure
                        x1 = existing_box['x1']
                        y1 = existing_box['y1']
                        class_name = existing_box.get('class_name', existing_box.get('name', ''))
                    else:
                        continue  # Skip invalid boxes
                        
                    sig = f"{class_name}-{round(x1)}-{round(y1)}"
                    existing_sigs.add(sig)
                
                # Process and add new boxes
                added_count = 0
                for box in boxes:
                    # IMPORTANT: Extract coordinates correctly based on structure
                    if 'box' in box and 'x1' in box['box'] and 'y1' in box['box']:
                        # Nested structure
                        x1 = box['box']['x1']
                        y1 = box['box']['y1']
                        class_name = box.get('class_name', '')
                    elif 'x1' in box and 'y1' in box:
                        # Flat structure (what we want)
                        x1 = box['x1']
                        y1 = box['y1']
                        class_name = box.get('class_name', box.get('name', ''))
                    else:
                        print(f"Skipping invalid box format: {box}")
                        continue
                        
                    # Create signature
                    sig = f"{class_name}-{round(x1)}-{round(y1)}"
                    
                    # Only add if it's not a duplicate
                    if sig not in existing_sigs:
                        # PRESERVE ALL FIELDS - add box exactly as received
                        combined_detections.append(box)
                        existing_sigs.add(sig)
                        added_count += 1
                        print(f"Added new box with signature: {sig}")
                
                print(f"Added {added_count} new boxes to combined detections")
            else:
                print("Warning: combined_detections is not a list, overwriting with new boxes")
                combined_detections = boxes
                
            # Save updated combined detections
            with open(combined_file, 'w') as f:
                json.dump(combined_detections, f, indent=2)
            print(f"Saved updated combined detections to: {combined_file}")
            
            # Validate the saved data to make sure it has all the expected fields
            try:
                with open(combined_file, 'r') as f:
                    saved_data = json.load(f)
                    print(f"Validated saved data - {len(saved_data)} boxes")
                    # Count boxes with orientation field
                    with_orientation = sum(1 for box in saved_data if 'orientation' in box)
                    print(f"Boxes with orientation: {with_orientation} of {len(saved_data)}")
            except Exception as e:
                print(f"Error validating saved data: {str(e)}")
            
        except Exception as e:
            print(f"Error processing combined detections: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Continue with response even if combined file fails

        return jsonify({
            'success': True,
            'message': 'Manual boxes saved successfully',
            'boxes_count': len(boxes),
            'wall_door_count': len(wall_door_boxes),
            'furniture_count': len(furniture_boxes),
            'saved_paths': {
                'manual_boxes': manual_boxes_file,
                'combined_detections': combined_file,
                'wall_door_manual': wall_door_file,
                'furniture_manual': furniture_file
            },
            'absolute_paths': {
                'manual_boxes': os.path.abspath(manual_boxes_file),
                'combined_detections': os.path.abspath(combined_file),
                'wall_door_manual': os.path.abspath(wall_door_file),
                'furniture_manual': os.path.abspath(furniture_file)
            },
            'debug_info': {
                'cwd': os.getcwd(),
                'output_dir': output_dir,
                'output_dir_exists': os.path.exists(output_dir),
                'output_dir_writable': os.access(output_dir, os.W_OK),
                'files_in_dir': os.listdir(output_dir) if os.path.exists(output_dir) else []
            }
        })

    except Exception as e:
        print(f"Error saving manual boxes: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({
            'error': str(e),
            'traceback': traceback_str,
            'debug_info': {
                'cwd': os.getcwd(),
                'output_folder_config': app.config['OUTPUT_FOLDER'],
                'abs_output_folder': os.path.abspath(app.config['OUTPUT_FOLDER']),
                'output_folder_exists': os.path.exists(os.path.abspath(app.config['OUTPUT_FOLDER'])),
                'output_folder_writable': os.access(os.path.abspath(app.config['OUTPUT_FOLDER']), os.W_OK)
            }
        }), 500

@app.route('/debug-file-path', methods=['GET'])
def debug_file_path():
    """Debug endpoint to check if a specific file exists and print its path"""
    try:
        session_id = request.args.get('session_id')
        file_type = request.args.get('file_type', 'combined_detections')
        
        if not session_id:
            return jsonify({
                "error": "No session_id provided",
                "status": "error"
            }), 400
        
        # Construct the expected file path
        file_name = f"{file_type}.json"
        relative_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, file_name)
        absolute_path = os.path.abspath(relative_path)
        
        # Check if the file exists
        file_exists = os.path.exists(absolute_path)
        
        # Get the file content and box count if it exists
        box_count = 0
        if file_exists:
            try:
                with open(absolute_path, 'r') as f:
                    file_content = json.load(f)
                    if isinstance(file_content, list):
                        box_count = len(file_content)
                    elif isinstance(file_content, dict) and 'boxes' in file_content:
                        box_count = len(file_content['boxes'])
            except Exception as e:
                print(f"Error reading file content: {str(e)}")
        
        # Check if the output directory exists
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        output_dir_exists = os.path.exists(output_dir)
        
        # If output directory doesn't exist, try to create it
        if not output_dir_exists:
            try:
                os.makedirs(output_dir, exist_ok=True)
                output_dir_exists = os.path.exists(output_dir)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Error creating output directory: {str(e)}")
        
        # List all files in the session directory if it exists
        files_in_dir = []
        if output_dir_exists:
            try:
                files_in_dir = os.listdir(output_dir)
            except Exception as e:
                print(f"Error listing directory: {str(e)}")
        
        # Return all the debug information
        return jsonify({
            "session_id": session_id,
            "file_type": file_type,
            "file_name": file_name,
            "relative_path": relative_path,
            "absolute_path": absolute_path,
            "file_exists": file_exists,
            "output_directory": output_dir,
            "output_directory_exists": output_dir_exists,
            "files_in_directory": files_in_dir,
            "box_count": box_count,
            "flask_root": app.root_path,
            "cwd": os.getcwd(),
            "output_folder_config": app.config['OUTPUT_FOLDER']
        })
    
    except Exception as e:
        import traceback
        print(f"Debug endpoint error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)