import os
import json
import cv2
import math
import numpy as np
import datetime
import uuid
import time
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for image dimensions
TARGET_WIDTH = 800
TARGET_HEIGHT = 600

class FurnitureDetectionProcessor:
    def __init__(self, base_dir, debug=False):
        # Directory structure
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.output_dir = os.path.join(base_dir, 'output')
        self.resized_dir = os.path.join(base_dir, 'static', 'resized')
        self.furniture_model_path = os.path.join(self.models_dir, 'furniture', 'weights1.pt')
        self.wall_door_model_path = os.path.join(self.models_dir, 'wall_door', 'weights2.pt')
        self.room_model_path = os.path.join(self.models_dir, 'room', 'weights3.pt')  # Add room model path
        self.debug = debug
     
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.resized_dir, exist_ok=True)
        
        # Initialize scale factors to None - will be set externally
        self.scale_factors = {
            "scale_x": None, 
            "scale_y": None
        }
        
        # Initialize models on demand to save memory
        self.furniture_model = None
        self.wall_door_model = None
        self.room_model = None  # Add room model
        
        # Define target dimensions as constants
        self.TARGET_WIDTH = 800
        self.TARGET_HEIGHT = 600
        self.wall_door_manual_boxes = []  # To store manual wall/door boxes
        self.furniture_manual_boxes = []  # To store manual furniture boxes
        self.aws_config = {
            'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('AWS_REGION', 'ap-south-1'),
            'bucket_name': os.getenv('AWS_S3_BUCKET', 'royaletouch'),
            'Objects_name': os.getenv('AWS_Objects', 'Furni4oraThemeThumb')
        }    
        
        # Validate required environment variables
        if not self.aws_config['access_key_id'] or not self.aws_config['secret_access_key']:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables")
 
    def load_room_model(self):
        """Load the room detection model on demand"""
        if self.room_model is None:
            from ultralytics import YOLO
            print(f"Loading room model from {self.room_model_path}")
            self.room_model = YOLO(self.room_model_path)
        return self.room_model
    
    def debug_model_output(self, results):
        """Debug function to print out the structure of model outputs"""
        if not self.debug:
            return
            
        print("Debugging model output structure:")
        
        # Check if results is iterable
        if not hasattr(results, '__iter__'):
            results = [results]
            
        for i, result in enumerate(results):
            print(f"Result {i}:")
            print(f"  Type: {type(result)}")
            print(f"  Available attributes: {dir(result)}")
            
            # Check for boxes
            if hasattr(result, 'boxes'):
                print(f"  Boxes type: {type(result.boxes)}")
                print(f"  Boxes attributes: {dir(result.boxes)}")
                
            # Check for keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                print(f"  Keypoints found!")
                print(f"  Keypoints type: {type(result.keypoints)}")
                print(f"  Keypoints attributes: {dir(result.keypoints)}")
                
                # Try to access keypoint data
                if hasattr(result.keypoints, 'data'):
                    print(f"  Keypoints data shape: {result.keypoints.data.shape}")
                    # Print an example keypoint if available
                    if len(result.keypoints.data) > 0:
                        print(f"  Example keypoint: {result.keypoints.data[0].cpu().numpy()}")
            else:
                print("  No keypoints found in this result.")
                
    def load_furniture_model(self):
        """Load the furniture detection model on demand"""
        if self.furniture_model is None:
            from ultralytics import YOLO
            print(f"Loading furniture model from {self.furniture_model_path}")
            self.furniture_model = YOLO(self.furniture_model_path)
        return self.furniture_model
    
    def load_wall_door_model(self):
        """Load the wall/door detection model on demand"""
        if self.wall_door_model is None:
            from ultralytics import YOLO
            print(f"Loading wall/door model from {self.wall_door_model_path}")
            self.wall_door_model = YOLO(self.wall_door_model_path)
        return self.wall_door_model
    
    def resize_image(self, image_path):
        """Resize image to exactly 800x600 preserving the entire image and aspect ratio"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Target dimensions
        target_width, target_height = self.TARGET_WIDTH, self.TARGET_HEIGHT
        
        # Get original dimensions
        img_height, img_width = img.shape[:2]
        
        # Check if image is already 800x600
        if img_width == target_width and img_height == target_height:
            # Create a copy in the resized folder for consistent processing
            unique_id = str(uuid.uuid4())
            resized_path = os.path.join(self.resized_dir, f"resized_{unique_id}.jpg")
            cv2.imwrite(resized_path, img)
            print(f"Image is already {target_width}x{target_height}, creating a copy in resized folder")
            return resized_path
        
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
        resized_path = os.path.join(self.resized_dir, f"resized_{unique_id}.jpg")
        cv2.imwrite(resized_path, resized)
        print(f"Saved resized image to: {resized_path}")
        
        return resized_path

    def ensure_image_size(self, image_path):
        """Ensure the image is exactly 800x600 pixels using INTER_AREA interpolation"""
        # Check if image is in the resized directory
        if 'resized_' in os.path.basename(image_path) and os.path.dirname(image_path) == self.resized_dir:
            # It's likely already resized, but verify dimensions
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Check dimensions
            img_height, img_width = img.shape[:2]
            if img_width == self.TARGET_WIDTH and img_height == self.TARGET_HEIGHT:
                print(f"Resized image verified with correct dimensions: {img_width}x{img_height}")
                return image_path
        
        # Either it's not in the resized directory or the dimensions are wrong
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Check dimensions
        img_height, img_width = img.shape[:2]
        if img_width == self.TARGET_WIDTH and img_height == self.TARGET_HEIGHT:
            # Correct dimensions but not in resized directory, create a copy there
            unique_id = str(uuid.uuid4())
            resized_path = os.path.join(self.resized_dir, f"resized_{unique_id}.jpg")
            cv2.imwrite(resized_path, img)
            print(f"Copied correctly sized image to resized directory: {resized_path}")
            return resized_path
            
        print(f"Image dimensions {img_width}x{img_height} are not {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}, resizing to resized directory...")
        return self.resize_image(image_path)
    
    def run_room_detection(self, image_path):
        """Run room detection model on the image"""
        # Ensure image is 800x600 and in the resized directory
        resized_path = self.ensure_image_size(image_path)
        
        # Log that we're using the resized image
        print(f"Running room detection on resized image: {resized_path}")
        
        # Load the model
        model = self.load_room_model()
        
        # Run inference
        results = model(resized_path, conf=0.01)
        
        # Debug the model output if debug is enabled
        if self.debug:
            self.debug_model_output(results)
        
        # Process results to match expected format
        predictions = []
        
        for result in results:
            # Process detection boxes
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                cls_name = result.names[cls_id]
                
                # Calculate center, width, height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Create the prediction object
                prediction = {
                    "class": cls_name,
                    "confidence": float(conf),
                    "x": float(center_x),
                    "y": float(center_y),
                    "width": float(width),
                    "height": float(height)
                }
                
                predictions.append(prediction)
        
        room_output = {"predictions": predictions}
        
        # Save the output
        unique_id = str(uuid.uuid4())
        room_output_path = os.path.join(self.output_dir, f"room_output_{unique_id}.json")
        with open(room_output_path, 'w') as f:
            json.dump(room_output, f, indent=4)
        
        # Return the resized_path and room output
        return resized_path, room_output

    def process_room_data(self, room_json_data, x_scale, y_scale):
        """Process room detection JSON data and apply scale factors"""
        output_data = {}
        
        for i, prediction in enumerate(room_json_data["predictions"], start=1):
            # Calculate corners from center and dimensions
            x, y = prediction["x"], prediction["y"]
            width, height = prediction["width"], prediction["height"]
            
            # Calculate corner coordinates (in clockwise order from top-left)
            corners = [
                {"x": x - width/2, "y": y - height/2},  # top-left
                {"x": x + width/2, "y": y - height/2},  # top-right
                {"x": x + width/2, "y": y + height/2},  # bottom-right
                {"x": x - width/2, "y": y + height/2}   # bottom-left
            ]
            
            # Apply scale factors to all coordinates
            scaled_x = prediction["x"] * x_scale
            scaled_y = prediction["y"] * y_scale
            scaled_width = prediction["width"] * x_scale
            scaled_height = prediction["height"] * y_scale
            
            scaled_corners = [
                {"x": corners[0]["x"] * x_scale, "y": corners[0]["y"] * y_scale},  # top-left
                {"x": corners[1]["x"] * x_scale, "y": corners[1]["y"] * y_scale},  # top-right
                {"x": corners[2]["x"] * x_scale, "y": corners[2]["y"] * y_scale},  # bottom-right
                {"x": corners[3]["x"] * x_scale, "y": corners[3]["y"] * y_scale}   # bottom-left
            ]
            
            # Create entry in output format
            output_data[f"A{i}"] = {
                "x": scaled_x,
                "y": scaled_y,
                "width": scaled_width,
                "height": scaled_height,
                "confidence": prediction["confidence"],
                "class": prediction["class"],
                "x1": scaled_corners[0]["x"],
                "y1": scaled_corners[0]["y"],
                "x2": scaled_corners[1]["x"],
                "y2": scaled_corners[1]["y"],
                "x3": scaled_corners[2]["x"],
                "y3": scaled_corners[2]["y"],
                "x4": scaled_corners[3]["x"],
                "y4": scaled_corners[3]["y"]
            }
        
        return output_data
    
    def run_furniture_detection(self, image_path):
        """Run furniture detection model on the image and extract proper keypoints"""
        # Ensure image is 800x600 and in the resized directory
        resized_path = self.ensure_image_size(image_path)
        
        # Log that we're using the resized image
        print(f"Running furniture detection on resized image: {resized_path}")
        
        # Load the model
        model = self.load_furniture_model()
        
        # Run inference
        results = model(resized_path,conf=0.01)
        
        # Debug the model output if debug is enabled
        if self.debug:
            self.debug_model_output(results)
        
        # Process results to match your expected format
        predictions = []
        
        try:
            # Check if we have keypoints in the standard format from Roboflow API
            if hasattr(results, 'json') and callable(getattr(results, 'json')):
                # If results has a JSON representation (like from Roboflow API)
                json_data = results.json()
                if 'predictions' in json_data:
                    return resized_path, json_data  # Return the JSON data directly
        except:
            pass  # Fall back to normal processing
        
        for result in results:
            # Process detection boxes
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                cls_name = result.names[cls_id]
                
                # Calculate center, width, height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Try to extract keypoints from the model output
                has_keypoints = False
                keypoints = []
                
                # Check if keypoints exist in the model output
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        # Try to access keypoints data
                        kpt_data = result.keypoints.data[i].cpu().numpy()
                        has_keypoints = True
                        
                        # Process the keypoints
                        for k in range(len(kpt_data)):
                            if len(kpt_data[k]) >= 3:
                                kpt_x, kpt_y, kpt_conf = kpt_data[k]
                            else:
                                kpt_x, kpt_y = kpt_data[k][:2]
                                kpt_conf = conf
                            
                            # Define keypoint class mappings for different furniture types
                            keypoint_mappings = {
                                "bed": {0: "bottom", 1: "top"},
                                "wardrobe": {0: "top", 1: "bottom"},
                                "door": {0: "bottom", 1: "top"},
                                "LargeSofa": {0: "bottom", 1: "top"},
                                "GasStove": {0: "top", 1: "bottom"},
                                "ToiletSeat": {0: "bottom", 1: "top"},
                                "WashBasin": {0: "bottom", 1: "top"},
                                "fridge": {0: "top", 1: "bottom"},
                                "ArmChair":{0:"bottom",1:"top"}
                                # Add mappings for other furniture types as needed
                            }                # Get the mapping for this furniture type (convert to lowercase for case-insensitive matching)
                            type_mapping = keypoint_mappings.get(cls_name, {0: "keypoint_0", 1: "keypoint_1"})
                            
                            # Get class name for this keypoint
                            kpt_class = type_mapping.get(k, f"keypoint_{k}")
                            
                            keypoints.append({
                                "x": float(kpt_x),
                                "y": float(kpt_y),
                                "confidence": float(kpt_conf),
                                "class_id": k,
                                "class": kpt_class
                            })
                    except Exception as e:
                        if self.debug:
                            print(f"Error extracting keypoints for {cls_name}: {str(e)}")
                        has_keypoints = False
                
                # If no keypoints were found, fall back to the bounding box method
                if not has_keypoints or len(keypoints) == 0:
                    if self.debug:
                        print(f"Using fallback keypoints for {cls_name}")
                    
                    # Use different keypoint assignments based on furniture type and orientation
                    if cls_name.lower() == "bed" or (width > height and cls_name.lower() not in ["wardrobe"]):
                        # For beds and horizontal furniture, use left-right keypoints
                        keypoints = [
                            {"x": float(x2), "y": float(center_y), "confidence": float(conf), "class_id": 0, "class": "bottom"},
                            {"x": float(x1), "y": float(center_y), "confidence": float(conf), "class_id": 1, "class": "top"}
                        ]
                    else:
                        # For wardrobes and vertical furniture, use top-bottom keypoints
                        keypoints = [
                            {"x": float(center_x), "y": float(y1), "confidence": float(conf), "class_id": 0, "class": "top"},
                            {"x": float(center_x), "y": float(y2), "confidence": float(conf), "class_id": 1, "class": "bottom"}
                        ]
                
                # Create the prediction object
                prediction = {
                    "class": cls_name,
                    "confidence": float(conf),
                    "x": float(center_x),
                    "y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                    "keypoints": keypoints
                }
                
                predictions.append(prediction)
        
        furniture_output = {"predictions": predictions}
        
        # Save the output
        unique_id = str(uuid.uuid4())
        furniture_output_path = os.path.join(self.output_dir, f"furniture_output_{unique_id}.json")
        with open(furniture_output_path, 'w') as f:
            json.dump(furniture_output, f, indent=4)
        
        # Return the resized_path (not the original) and furniture output
        return resized_path, furniture_output
        
    def run_wall_door_detection(self, image_path):
        """Run wall/door detection model on the image and extract proper keypoints"""
        # Ensure image is 800x600 before running model
        resized_path = self.ensure_image_size(image_path)
        
        # Log that we're using the resized image
        print(f"Running wall/door detection on resized image: {resized_path}")
        
        # Load the model
        model = self.load_wall_door_model()
        
        # Run inference
        results = model(resized_path,conf=0.01)
        
        # Debug the model output if debug is enabled
        if self.debug:
            self.debug_model_output(results)
        
        # Process results to match your expected format
        predictions = []
        
        try:
            # Check if we have keypoints in the standard format from Roboflow API
            if hasattr(results, 'json') and callable(getattr(results, 'json')):
                # If results has a JSON representation (like from Roboflow API)
                json_data = results.json()
                if 'predictions' in json_data:
                    return resized_path, json_data  # Return the JSON data directly
        except:
            pass  # Fall back to normal processing
        
        for result in results:
            # Process detection boxes
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                cls_name = result.names[cls_id]
                
                # Calculate center, width, height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Try to extract keypoints from the model output
                has_keypoints = False
                keypoints = []
                
                # Check if keypoints exist in the model output
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        # Try to access keypoints data
                        kpt_data = result.keypoints.data[i].cpu().numpy()
                        has_keypoints = True
                        
                        # Process the keypoints
                        for k in range(len(kpt_data)):
                            if len(kpt_data[k]) >= 3:
                                kpt_x, kpt_y, kpt_conf = kpt_data[k]
                            else:
                                kpt_x, kpt_y = kpt_data[k][:2]
                                kpt_conf = conf
                            
                            # Define keypoint class mappings for different wall/door types
                            keypoint_mappings = {
                                "wall": {0: "wall"},  # Wall typically has one keypoint labeled 'wall'
                                "door": {0: "bottom", 1: "top"},
                                "window": {0: "window"},  # Window typically has one keypoint labeled 'window'
                                "slider": {0: "window"},
                                # Add more mappings as needed for wall/door types
                            }
                            
                            # Get the mapping for this wall/door type (convert to lowercase for case-insensitive matching)
                            type_mapping = keypoint_mappings.get(cls_name, {0: "keypoint_0", 1: "keypoint_1"})
                            
                            # Get class name for this keypoint
                            kpt_class = type_mapping.get(k, f"keypoint_{k}")
                            
                            keypoints.append({
                                "x": float(kpt_x),
                                "y": float(kpt_y),
                                "confidence": float(kpt_conf),
                                "class_id": k,
                                "class": kpt_class
                            })
                    except Exception as e:
                        if self.debug:
                            print(f"Error extracting keypoints for {cls_name}: {str(e)}")
                        has_keypoints = False
                
                # If no keypoints were found, fall back to the bounding box method
                if not has_keypoints or len(keypoints) == 0:
                    if self.debug:
                        print(f"Using fallback keypoints for {cls_name}")
                    
                    # Use different keypoint assignments based on wall/door type and orientation
                    if cls_name.lower() == "wall" or cls_name.lower() == "window":
                        # For horizontal walls/windows
                        if width > height:
                            keypoints = [
                                {"x": float(x1), "y": float(center_y), "confidence": float(conf), "class_id": 0, "class": "wall"},
                                {"x": float(x2), "y": float(center_y), "confidence": float(conf), "class_id": 1, "class": "wall"}
                            ]
                        # For vertical walls/windows
                        else:
                            keypoints = [
                                {"x": float(center_x), "y": float(y1), "confidence": float(conf), "class_id": 0, "class": "wall"},
                                {"x": float(center_x), "y": float(y2), "confidence": float(conf), "class_id": 1, "class": "wall"}
                            ]
                    elif cls_name.lower() == "door":
                        # For doors, use top/bottom based on orientation
                        if height > width:  # Vertical door
                            keypoints = [
                                {"x": float(center_x), "y": float(y1), "confidence": float(conf), "class_id": 1, "class": "top"},
                                {"x": float(center_x), "y": float(y2), "confidence": float(conf), "class_id": 0, "class": "bottom"}
                            ]
                        else:  # Horizontal door
                            keypoints = [
                                {"x": float(x1), "y": float(center_y), "confidence": float(conf), "class_id": 1, "class": "top"},
                                {"x": float(x2), "y": float(center_y), "confidence": float(conf), "class_id": 0, "class": "bottom"}
                            ]
                    else:
                        # For other elements, default to orientation-based keypoints
                        if height > width:
                            keypoints = [
                                {"x": float(center_x), "y": float(y1), "confidence": float(conf), "class_id": 0, "class": "top"},
                                {"x": float(center_x), "y": float(y2), "confidence": float(conf), "class_id": 1, "class": "bottom"}
                            ]
                        else:
                            keypoints = [
                                {"x": float(x1), "y": float(center_y), "confidence": float(conf), "class_id": 0, "class": "top"},
                                {"x": float(x2), "y": float(center_y), "confidence": float(conf), "class_id": 1, "class": "bottom"}
                            ]
                
                # Create the prediction object
                prediction = {
                    "class": cls_name,
                    "confidence": float(conf),
                    "x": float(center_x),
                    "y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                    "keypoints": keypoints
                }
                
                predictions.append(prediction)
        
        wall_door_output = {"predictions": predictions}
        
        # Save the output
        unique_id = str(uuid.uuid4())
        wall_door_output_path = os.path.join(self.output_dir, f"wall_door_output_{unique_id}.json")
        with open(wall_door_output_path, 'w') as f:
            json.dump(wall_door_output, f, indent=4)
        
        # Return the resized_path (not the original) and wall_door output
        return resized_path, wall_door_output

    def process_image(self, image_path, confidence_threshold=0.01, session_id=None, include_manual_boxes=False):
     """Main entry point for processing an image with optional confidence threshold"""
     try:
        print(f"Processing image: {image_path} with confidence threshold: {confidence_threshold}")
        
        # First, ensure the image is exactly 800x600
        resized_path = self.ensure_image_size(image_path)
        print(f"Using resized image: {resized_path}")
        
        # Extract session_id from path if not provided
        if session_id is None:
            # Try to extract session ID from path
            try:
                path_parts = Path(image_path).parts
                for part in path_parts:
                    if '-' in part and len(part) > 30:
                        # Looks like a UUID
                        session_id = part
                        break
            except:
                pass
                
        # Validate scale factors
        if not self.scale_factors or self.scale_factors["scale_x"] is None or self.scale_factors["scale_y"] is None:
            raise ValueError("Scale factors must be set before processing the image")
        
        x_scale = self.scale_factors["scale_x"]
        y_scale = self.scale_factors["scale_y"]
        
        print(f"Using scale factors - X: {x_scale}, Y: {y_scale}")
        
        # If manual boxes should be included, load them first
        manual_boxes_added = False
        if include_manual_boxes and session_id:
            # Load the manual boxes from the session folder
            manual_boxes_added = self.load_manual_boxes(session_id)
            print(f"Manual boxes loaded: {len(self.wall_door_manual_boxes)} wall/door, {len(self.furniture_manual_boxes)} furniture")
        
        # 1. Run furniture detection on the resized image
        start_time = time.time()
        _, furniture_data = self.run_furniture_detection(resized_path)
        print(f"Furniture detection completed in {time.time() - start_time:.2f}s")
        
        # Apply confidence threshold to furniture data
        if 'predictions' in furniture_data:
            original_count = len(furniture_data['predictions'])
            furniture_data['predictions'] = [
                pred for pred in furniture_data['predictions'] 
                if pred.get('confidence', 0) >= confidence_threshold
            ]
            print(f"Applied confidence threshold {confidence_threshold}: Filtered furniture detections from {original_count} to {len(furniture_data['predictions'])}")
        
        # Save the original furniture data for reference
        furniture_output_path = os.path.join(self.output_dir, f"furniture_output_{str(uuid.uuid4())}.json")
        with open(furniture_output_path, 'w') as f:
            json.dump(furniture_data, f, indent=4)
        print(f"Saved original furniture data: {furniture_output_path}")
        
        # 2. Process furniture data with scaling
        start_time = time.time()
        furniture_processed = self.process_furniture_data(furniture_data, x_scale, y_scale)
        
        # Add manual furniture boxes if available
        if include_manual_boxes and self.furniture_manual_boxes:
            self.add_manual_boxes_to_json(furniture_processed, self.furniture_manual_boxes, "furniture")
            print(f"Added {len(self.furniture_manual_boxes)} manual furniture boxes to FURNITURE.json")
        
        # Save processed furniture data
        furniture_final_path = os.path.join(self.output_dir, "FURNITURE.json")
        with open(furniture_final_path, 'w') as f:
            json.dump(furniture_processed, f, indent=4)
        print(f"Furniture data processed in {time.time() - start_time:.2f}s: {furniture_final_path}")
        
        # 3. Run wall/door detection on the resized image
        start_time = time.time()
        _, wall_door_data = self.run_wall_door_detection(resized_path)
        print(f"Wall/door detection completed in {time.time() - start_time:.2f}s")
        
        # Apply confidence threshold to wall/door data
        if 'predictions' in wall_door_data:
            original_count = len(wall_door_data['predictions'])
            wall_door_data['predictions'] = [
                pred for pred in wall_door_data['predictions'] 
                if pred.get('confidence', 0) >= confidence_threshold
            ]
            print(f"Applied confidence threshold {confidence_threshold}: Filtered wall/door detections from {original_count} to {len(wall_door_data['predictions'])}")
        
        # Save the original wall/door data for reference
        wall_door_output_path = os.path.join(self.output_dir, f"wall_door_output_{str(uuid.uuid4())}.json")
        with open(wall_door_output_path, 'w') as f:
            json.dump(wall_door_data, f, indent=4)
        print(f"Saved original wall/door data: {wall_door_output_path}")
        
        # 4. Process wall/door data
        start_time = time.time()
        wall_door_processed = self.process_wall_door_data(wall_door_data, x_scale, y_scale)
        
        # Add manual wall/door boxes if available
        if include_manual_boxes and self.wall_door_manual_boxes:
            self.add_manual_boxes_to_json(wall_door_processed, self.wall_door_manual_boxes, "wall_door")
            print(f"Added {len(self.wall_door_manual_boxes)} manual wall/door boxes to WALL.json")
        
        # Save final processed data
        final_output_path = os.path.join(self.output_dir, "WALL.json")
        with open(final_output_path, 'w') as f:
            json.dump(wall_door_processed, f, indent=4)
        print(f"Wall/door data processed in {time.time() - start_time:.2f}s: {final_output_path}")
        
        # 5. Run room detection on the resized image
        start_time = time.time()
        _, room_data = self.run_room_detection(resized_path)
        print(f"Room detection completed in {time.time() - start_time:.2f}s")
        
        # Apply confidence threshold to room data
        if 'predictions' in room_data:
            original_count = len(room_data['predictions'])
            room_data['predictions'] = [
                pred for pred in room_data['predictions'] 
                if pred.get('confidence', 0) >= confidence_threshold
            ]
            print(f"Applied confidence threshold {confidence_threshold}: Filtered room detections from {original_count} to {len(room_data['predictions'])}")
        
        # Save the original room data for reference
        room_output_path = os.path.join(self.output_dir, f"room_output_{str(uuid.uuid4())}.json")
        with open(room_output_path, 'w') as f:
            json.dump(room_data, f, indent=4)
        print(f"Saved original room data: {room_output_path}")
        
        # 6. Process room data with scaling
        start_time = time.time()
        room_processed = self.process_room_data(room_data, x_scale, y_scale)
        
        # Save processed room data
        room_final_path = os.path.join(self.output_dir, "ROOM.json")
        with open(room_final_path, 'w') as f:
            json.dump(room_processed, f, indent=4)
        print(f"Room data processed in {time.time() - start_time:.2f}s: {room_final_path}")
        
        # 7. Upload files to AWS S3
        start_time = time.time()
        furniture_s3_path = self.upload_to_aws_s3(furniture_final_path, "FURNITURE.json")
        wall_s3_path = self.upload_to_aws_s3(final_output_path, "WALL.json")
        room_s3_path = self.upload_to_aws_s3(room_final_path, "ROOM.json")
        print(f"Files uploaded to AWS in {time.time() - start_time:.2f}s")
        print(f"Furniture S3 path: {furniture_s3_path}")
        print(f"Wall S3 path: {wall_s3_path}")
        print(f"Room S3 path: {room_s3_path}")
        
        # 8. Combine results for the frontend
        combined_results = {
            "furniture": furniture_processed,
            "wall_door": wall_door_processed,
            "room": room_processed,  # Add room data
            "image_path": resized_path,  # Return the path to the resized image
            "unscaled_detections": self.format_unscaled_detections(furniture_data, wall_door_data),  # Add unscaled detections
            "furniture_predictions": furniture_data,  # Add the filtered furniture predictions for reference
            "wall_door_predictions": wall_door_data,  # Add the filtered wall/door predictions for reference
            "room_predictions": room_data,  # Add the filtered room predictions for reference
            "confidence_threshold": confidence_threshold,  # Include the applied confidence threshold
            "aws_paths": {
                "furniture": furniture_s3_path,
                "wall": wall_s3_path,
                "room": room_s3_path  # Add room S3 path
            },
            "manual_boxes_added": manual_boxes_added  # Flag to indicate if manual boxes were added
        }
        
        return combined_results
        
     except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise
    
    def upload_to_aws_s3(self, local_file_path, s3_file_name=None):
        """
        Uploads a file to AWS S3 using secure protocols and makes it publicly readable
        
        Args:
            local_file_path (str): Path to the local file to upload
            s3_file_name (str, optional): Name to use for the file in S3. 
                                         If None, uses the basename of local_file_path
        
        Returns:
            str: S3 URI of the uploaded file
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            import datetime
            
            # Get the S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_config.get('access_key_id'),
                aws_secret_access_key=self.aws_config.get('secret_access_key'),
                region_name=self.aws_config.get('region_name')
            )
            
            # Determine bucket and file name
            bucket_name = self.aws_config.get('bucket_name')
            objects_folder = self.aws_config.get('Objects_name')
            
            if s3_file_name is None:
                s3_file_name = os.path.basename(local_file_path)
            
            # Use fixed path without timestamps for overwriting
            s3_key = f"{objects_folder}/{s3_file_name}"
            
            # Upload the file with server-side encryption and public-read ACL
            s3_client.upload_file(
                local_file_path, 
                bucket_name, 
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',  # Enable server-side encryption
                    'ContentType': 'application/json',  # Set proper content type
                    'ACL': 'public-read'  # Make object publicly readable
                }
            )
            
            # Return the S3 URI and also the public URL
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            public_url = f"https://{bucket_name}.s3.{self.aws_config.get('region_name')}.amazonaws.com/{s3_key}"
            
            print(f"Public URL: {public_url}")
            
            return s3_uri
            
        except ImportError:
            print("WARNING: boto3 not installed. Install with 'pip install boto3' to enable AWS uploads.")
            return None
        except ClientError as e:
            print(f"AWS S3 upload error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            return None

    def format_unscaled_detections(self, furniture_data, wall_door_data):
        """Format unscaled detection results for displaying bounding boxes"""
        detections = []
        
        # Process furniture detections
        if 'predictions' in furniture_data:
            for prediction in furniture_data['predictions']:
                # Use the center-based coordinates from YOLO's raw output
                detections.append({
                    "class_name": prediction["class"],
                    "type": "furniture",
                    "confidence": prediction["confidence"],
                    "box": {
                        # Calculate corner coordinates from center-based YOLO output
                        "x1": prediction["x"] - prediction["width"] / 2,
                        "y1": prediction["y"] - prediction["height"] / 2,
                        "width": prediction["width"],
                        "height": prediction["height"]
                    }
                })
        
        # Process wall/door detections
        if 'predictions' in wall_door_data:
            for prediction in wall_door_data['predictions']:
                detections.append({
                    "class_name": prediction["class"],
                    "type": "wall_door",
                    "confidence": prediction["confidence"],
                    "box": {
                        # Calculate corner coordinates from center-based YOLO output
                        "x1": prediction["x"] - prediction["width"] / 2,
                        "y1": prediction["y"] - prediction["height"] / 2,
                        "width": prediction["width"],
                        "height": prediction["height"]
                    }
                })
        
        return detections
    
    def process_furniture_data(self, data, x_scale, y_scale):
        """Process furniture detection results"""
        # Rest of the method remains the same as in the previous implementation
        processed_data = {}
        
        # A1: Center point and scale factors
        center_x = (800 / 2) * x_scale
        center_y = (600 / 2) * y_scale
        
        processed_data["A1"] = {
            "center_x": center_x,
            "center_y": center_y,
            "x_scale": x_scale,
            "y_scale": y_scale
        }
        
        # Process furniture data
        for i, obj in enumerate(data["predictions"], start=2):
            width = obj["width"] * x_scale
            height = obj["height"] * y_scale
            x = obj["x"] * x_scale
            y = obj["y"] * y_scale
            
            if width >= height or width == height:
                x1 = x - width / 2
                x2 = x + width / 2
                y1 = y
                y2 = y
            else:
                x1 = x
                x2 = x
                y1 = y - height / 2
                y2 = y + height / 2
            
            slope, angle, facing, top, bottom = self.calculate_slope_and_angle(obj["keypoints"])
            
            orientation = "Vertical" if obj["height"] > obj["width"] else "Horizontal"
            
            processed_obj = {
                "name": obj["class"],
                "confidence": obj["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "TopX": top["x"] * x_scale if top else None,
                "TopY": top["y"] * y_scale if top else None,
                "BottomX": bottom["x"] * x_scale if bottom else None,
                "BottomY": bottom["y"] * y_scale if bottom else None,
                "width": width,
                "height": height,
                "facing": facing,
                "slope": str(slope) if slope is not None else "N/A",
                "angle": str(angle) if angle is not None else "N/A",
                "orientation": orientation,
                "keypoints": [
                    {
                        "x": kp["x"] * x_scale,
                        "y": kp["y"] * y_scale,
                        "confidence": kp["confidence"],
                        "class_id": kp["class_id"],
                        "class_name": kp["class"]
                    } for kp in obj["keypoints"]
                ]
            }
            
            processed_data[f"A{i}"] = processed_obj
        
        return processed_data
    
    # ... (rest of the methods remain the same)
    
    def process_wall_door_data(self, data, x_scale, y_scale, threshold_height=130, threshold_width=130):
        """Process wall/door detection results"""
        # First process similar to furniture processing
        processed_data = self.process_furniture_data(data, x_scale, y_scale)
        
        # Then separate into wall, door, slider categories
        wall_data = {}
        slider_data = {}
        window_data = {}
        
        # Copy A1 (metadata)
        wall_data["A1"] = processed_data["A1"]
        
        # Process each object
        for i, obj in enumerate(list(processed_data.items())[1:], start=2):
            key, obj_data = obj
            
            if isinstance(obj_data, dict) and "name" in obj_data:
                width = obj_data["width"]
                height = obj_data["height"]
                orientation = obj_data["orientation"]
                
                converted_to_slider = False
                
                if obj_data["name"].lower() == "window":
                    if orientation == "Vertical" and height > threshold_height:
                        obj_data["name"] = "slider"
                        converted_to_slider = True
                    elif orientation == "Horizontal" and width > threshold_width:
                        obj_data["name"] = "slider"
                        converted_to_slider = True
                
                if converted_to_slider:
                    slider_data[key] = obj_data
                else:
                    window_data[key] = obj_data
                
                # Include walls, doors, and converted sliders in wall_data
                if obj_data["name"].lower() in ["wall", "door", "slider"]:
                    wall_data[key] = obj_data
        
        # Include non-converted windows
        for key, value in window_data.items():
            if value["name"].lower() == "window":
                wall_data[key] = value
        
        # Apply door coordinate updates
        wall_data = self.update_door_coordinates(wall_data, threshold=15)
        
        return wall_data
    
    def calculate_slope_and_angle(self, keypoints):
        """Calculate slope and angle between keypoints"""
        top = None
        bottom = None
        
        for kp in keypoints:
            if kp["class"] == "top":
                top = kp
            elif kp["class"] == "bottom":
                bottom = kp
        
        if top and bottom:
            dx = abs(top["x"] - bottom["x"])
            dy = abs(top["y"] - bottom["y"])
            
            if dx == 0:
                slope = float('inf')  # Vertical line
            else:
                slope = round((top["y"] - bottom["y"]) / (top["x"] - bottom["x"]), 2)
            
            angle = round(math.degrees(math.atan2(top["y"] - bottom["y"], top["x"] - bottom["x"])), 2)
            
            if dy > dx:
                if bottom["y"] > top["y"]:
                    facing = "Downwards facing"
                else:
                    facing = "Upwards facing"
            else:
                if top["x"] > bottom["x"]:
                    facing = "Leftside facing"
                else:
                    facing = "Rightside facing"
            
            return slope, angle, facing, top, bottom
        
        return None, None, "Facing not available", None, None
    
    def chessboard_distance(self, x1, y1, x2, y2):
        """Calculate chessboard (Chebyshev) distance between two points"""
        return max(abs(x2 - x1), abs(y2 - y1))
    
    def update_door_coordinates(self, data, threshold):
        """Update door coordinates based on nearby walls/windows/sliders"""
        for key, obj in data.items():
            if isinstance(obj, dict) and obj.get("name") == "door":
                min_top_distance = float('inf')
                min_bottom_distance = float('inf')
                new_top_x, new_top_y = obj.get("TopX"), obj.get("TopY")
                new_bottom_x, new_bottom_y = obj.get("BottomX"), obj.get("BottomY")
                
                for target_key, target_obj in data.items():
                    if isinstance(target_obj, dict) and target_obj.get("name") in ["wall", "window", "slider"]:
                        # Compare door top coordinates with target coordinates
                        if obj.get("TopX") is not None and obj.get("TopY") is not None:
                            top_distance_1 = self.chessboard_distance(
                                obj["TopX"], obj["TopY"], target_obj["x1"], target_obj["y1"]
                            )
                            top_distance_2 = self.chessboard_distance(
                                obj["TopX"], obj["TopY"], target_obj["x2"], target_obj["y2"]
                            )

                            # Update top coordinates if a closer object within threshold is found
                            if top_distance_1 < threshold and top_distance_1 < min_top_distance:
                                min_top_distance = top_distance_1
                                new_top_x, new_top_y = target_obj["x1"], target_obj["y1"]
                            if top_distance_2 < threshold and top_distance_2 < min_top_distance:
                                min_top_distance = top_distance_2
                                new_top_x, new_top_y = target_obj["x2"], target_obj["y2"]

                        # Compare door bottom coordinates with target coordinates
                        if obj.get("BottomX") is not None and obj.get("BottomY") is not None:
                            bottom_distance_1 = self.chessboard_distance(
                                obj["BottomX"], obj["BottomY"], target_obj["x1"], target_obj["y1"]
                            )
                            bottom_distance_2 = self.chessboard_distance(
                                obj["BottomX"], obj["BottomY"], target_obj["x2"], target_obj["y2"]
                            )

                            # Update bottom coordinates if a closer object within threshold is found
                            if bottom_distance_1 < threshold and bottom_distance_1 < min_bottom_distance:
                                min_bottom_distance = bottom_distance_1
                                new_bottom_x, new_bottom_y = target_obj["x1"], target_obj["y1"]
                            if bottom_distance_2 < threshold and bottom_distance_2 < min_bottom_distance:
                                min_bottom_distance = bottom_distance_2
                                new_bottom_x, new_bottom_y = target_obj["x2"], target_obj["y2"]

                # Update door's coordinates only if they have been adjusted within the threshold
                if min_top_distance < float('inf'):
                    obj["TopX"], obj["TopY"] = new_top_x, new_top_y
                if min_bottom_distance < float('inf'):
                    obj["BottomX"], obj["BottomY"] = new_bottom_x, new_bottom_y
        
        return data

    def load_manual_boxes(self, session_id):
     """
    Load manually created boxes from manual_boxes.json and separate them by type
    
    Args:
        session_id (str): Session ID to use for finding manual boxes
    
    Returns:
        bool: True if manual boxes were loaded, False otherwise
     """
     try:
        # Clear existing manual boxes
        self.wall_door_manual_boxes = []
        self.furniture_manual_boxes = []
        
        # Look for manual_boxes.json in the session directory
        manual_boxes_path = os.path.join(self.output_dir, session_id, 'manual_boxes.json')
        
        if not os.path.exists(manual_boxes_path):
            print(f"No manual_boxes.json found at {manual_boxes_path}")
            
            # Also check for separate wall_door_manual.json and furniture_manual.json
            wall_door_manual_path = os.path.join(self.output_dir, session_id, 'wall_door_manual.json')
            furniture_manual_path = os.path.join(self.output_dir, session_id, 'furniture_manual.json')
            
            if os.path.exists(wall_door_manual_path):
                try:
                    with open(wall_door_manual_path, 'r') as f:
                        self.wall_door_manual_boxes = json.load(f)
                    print(f"Loaded {len(self.wall_door_manual_boxes)} wall/door manual boxes from {wall_door_manual_path}")
                except Exception as e:
                    print(f"Error loading wall_door_manual.json: {str(e)}")
            
            if os.path.exists(furniture_manual_path):
                try:
                    with open(furniture_manual_path, 'r') as f:
                        self.furniture_manual_boxes = json.load(f)
                    print(f"Loaded {len(self.furniture_manual_boxes)} furniture manual boxes from {furniture_manual_path}")
                except Exception as e:
                    print(f"Error loading furniture_manual.json: {str(e)}")
            
            return len(self.wall_door_manual_boxes) > 0 or len(self.furniture_manual_boxes) > 0
        
        # Load the manual boxes
        with open(manual_boxes_path, 'r') as f:
            manual_boxes = json.load(f)
        
        if not manual_boxes:
            print("No manual boxes found in file")
            return False
        
        print(f"Found {len(manual_boxes)} manual boxes, separating by type")
        
        # Separate boxes by type
        for box in manual_boxes:
            box_type = box.get('type', '').lower()
            
            if box_type == 'wall_door':
                self.wall_door_manual_boxes.append(box)
            elif box_type == 'furniture':
                self.furniture_manual_boxes.append(box)
            else:
                print(f"Unknown box type: {box_type}, defaulting to wall_door")
                # Default to wall_door if type is not specified
                self.wall_door_manual_boxes.append(box)
        
        print(f"Separated {len(self.wall_door_manual_boxes)} wall/door boxes and {len(self.furniture_manual_boxes)} furniture boxes")
        
        # Save the separated boxes for future use
        os.makedirs(os.path.join(self.output_dir, session_id), exist_ok=True)
        
        wall_door_manual_path = os.path.join(self.output_dir, session_id, 'wall_door_manual.json')
        with open(wall_door_manual_path, 'w') as f:
            json.dump(self.wall_door_manual_boxes, f, indent=4)
        
        furniture_manual_path = os.path.join(self.output_dir, session_id, 'furniture_manual.json')
        with open(furniture_manual_path, 'w') as f:
            json.dump(self.furniture_manual_boxes, f, indent=4)
        
        return len(manual_boxes) > 0
    
     except Exception as e:
        import traceback
        print(f"Error loading manual boxes: {str(e)}")
        print(traceback.format_exc())
        return False

# Add this new method to add manual boxes to the appropriate JSON file
    def add_manual_boxes_to_json(self, json_data, manual_boxes, box_type):
     """
    Add manually created boxes to the appropriate JSON file
    
    Args:
        json_data (dict): The JSON data to add the boxes to
        manual_boxes (list): List of manual boxes to add
        box_type (str): Type of boxes ('furniture' or 'wall_door')
    
    Returns:
        int: Number of boxes added
     """
     try:
        if not manual_boxes:
            return 0
        
        # Get scale factors for converting coordinates
        x_scale = self.scale_factors.get("scale_x", 1.0)
        y_scale = self.scale_factors.get("scale_y", 1.0)
        
        print(f"DEBUG: Using scale factors x_scale={x_scale}, y_scale={y_scale}")
        
        # Determine the highest existing "A" key number
        highest_a_num = 1  # Start with A1 which is always present
        for key in json_data.keys():
            if key.startswith('A') and key[1:].split('_')[0].isdigit():
                try:
                    num = int(key[1:].split('_')[0])
                    if num > highest_a_num:
                        highest_a_num = num
                except ValueError:
                    pass
        
        next_a_new_num = 1
        added_count = 0
        
        for box in manual_boxes:
            print(f"DEBUG: Processing manual box: {json.dumps(box, indent=2)}")
            
            # Skip boxes we've already processed (check by coordinates to avoid duplicates)
            duplicate = False
            for key, value in json_data.items():
                if isinstance(value, dict) and 'x1' in value and 'y1' in value:
                    # Check for close coordinates (within 5 pixels)
                    box_x1 = box.get('x1', 0)
                    box_y1 = box.get('y1', 0)
                    
                    # Handle both flat and nested structure
                    if 'box' in box:
                        box_x1 = box['box'].get('x1', 0)
                        box_y1 = box['box'].get('y1', 0)
                    
                    if (abs(value.get('x1', 0) - box_x1) < 5 and 
                        abs(value.get('y1', 0) - box_y1) < 5):
                        duplicate = True
                        break
            
            if duplicate:
                print(f"Skipping duplicate box: {box.get('class_name', box.get('name', ''))} at x1={box_x1}, y1={box_y1}")
                continue
            
            # Format for JSON (similar to process_furniture_data output)
            class_name = box.get('class_name') or box.get('name', "unknown")
            
            # Get box coordinates, handling both flat and nested structure
            x1, y1, x2, y2 = 0, 0, 0, 0
            width, height = 0, 0
            
            # IMPORTANT: The coordinates from frontend are already scaled!
            # But we need to check if they need to be scaled again based on the context
            if 'box' in box:
                # Nested structure
                x1 = box['box'].get('x1', 0)
                y1 = box['box'].get('y1', 0)
                width = box['box'].get('width', 0)
                height = box['box'].get('height', 0)
                x2 = x1 + width
                y2 = y1 + height
            else:
                # Flat structure - these are already scaled coordinates from frontend
                x1 = box.get('x1', 0)
                y1 = box.get('y1', 0)
                x2 = box.get('x2', 0)
                y2 = box.get('y2', 0)
                width = box.get('width', 0)
                height = box.get('height', 0)
                
                # Calculate missing values if needed
                if width == 0 and x1 != x2:
                    width = abs(x2 - x1)
                if height == 0 and y1 != y2:
                    height = abs(y2 - y1)
            
            print(f"DEBUG: Box coordinates - x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={width}, height={height}")
            
            # Process keypoints for TopX, TopY, BottomX, BottomY
            # IMPORTANT: These coordinates are from frontend and may need scaling
            top_x, top_y = None, None
            bottom_x, bottom_y = None, None
            processed_keypoints = []
            
            # Check if keypoints are provided in the keypoints array
            if 'keypoints' in box and box['keypoints'] and len(box['keypoints']) > 0:
                print(f"DEBUG: Found {len(box['keypoints'])} keypoints in box")
                for kp in box['keypoints']:
                    kp_type = kp.get('type')
                    # These coordinates are already from the scaled backend data
                    # NO NEED TO SCALE AGAIN - they're already in real-world coordinates
                    kp_x = kp.get('x', 0)
                    kp_y = kp.get('y', 0)
                    
                    print(f"DEBUG: Keypoint {kp_type}: coordinates=({kp_x}, {kp_y})")
                    
                    # Store keypoint coordinates
                    if kp_type == 'top':
                        top_x = kp_x
                        top_y = kp_y
                    elif kp_type == 'bottom':
                        bottom_x = kp_x
                        bottom_y = kp_y
                    
                    # Add to processed keypoints
                    processed_keypoints.append({
                        "x": kp_x,
                        "y": kp_y,
                        "confidence": 1.0,
                        "class_id": 0 if kp_type == 'top' else 1,
                        "class": kp_type
                    })
            
            # Also check if TopX, TopY are directly defined (these are already scaled)
            if 'TopX' in box and box['TopX'] is not None:
                top_x = box['TopX']
                    
            if 'TopY' in box and box['TopY'] is not None:
                top_y = box['TopY']
                    
            if 'BottomX' in box and box['BottomX'] is not None:
                bottom_x = box['BottomX']
                    
            if 'BottomY' in box and box['BottomY'] is not None:
                bottom_y = box['BottomY']
            
            print(f"DEBUG: Final keypoint coordinates - Top: ({top_x}, {top_y}), Bottom: ({bottom_x}, {bottom_y})")
            
            formatted_box = {
                "name": class_name,
                "confidence": box.get('confidence', 1.0),
                "x1": x1,  # These are already scaled from frontend
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "TopX": top_x,  # Use processed keypoint coordinates (scaled)
                "TopY": top_y,
                "BottomX": bottom_x,
                "BottomY": bottom_y,
                "width": width,
                "height": height,
                "facing": box.get('facing', "Manually drawn"),
                "slope": box.get('slope', "N/A"),
                "angle": box.get('angle', "N/A"),
                "orientation": box.get('orientation', "Horizontal" if width >= height else "Vertical"),
                "keypoints": processed_keypoints,  # Add processed keypoints
                "manual": True  # Mark as manual box
            }
            
            # Add center coordinates if available
            if 'x' in box and 'y' in box:
                formatted_box["x"] = box['x']  # These should already be scaled
                formatted_box["y"] = box['y']
            else:
                # Calculate center from x1, y1, width, height
                formatted_box["x"] = x1 + width / 2
                formatted_box["y"] = y1 + height / 2
            
            # Calculate facing direction from keypoints if available
            if top_x is not None and top_y is not None and bottom_x is not None and bottom_y is not None:
                dx = abs(top_x - bottom_x)
                dy = abs(top_y - bottom_y)
                
                if dy > dx:
                    if bottom_y > top_y:
                        formatted_box["facing"] = "Downwards facing"
                    else:
                        formatted_box["facing"] = "Upwards facing"
                else:
                    if top_x > bottom_x:
                        formatted_box["facing"] = "Leftside facing"
                    else:
                        formatted_box["facing"] = "Rightside facing"
                
                # Calculate slope and angle
                if dx == 0:
                    formatted_box["slope"] = str(float('inf'))  # Vertical line
                else:
                    formatted_box["slope"] = str(round((top_y - bottom_y) / (top_x - bottom_x), 2))
                
                angle = round(math.degrees(math.atan2(top_y - bottom_y, top_x - bottom_x)), 2)
                formatted_box["angle"] = str(angle)
            
            # Add to json_data with a new key
            new_key = f"A{highest_a_num + next_a_new_num}_manual"
            json_data[new_key] = formatted_box
            next_a_new_num += 1
            added_count += 1
            print(f"Added manual {box_type} box as {new_key}: {formatted_box['name']}")
            print(f"DEBUG: Added box with TopX={formatted_box['TopX']}, TopY={formatted_box['TopY']}, BottomX={formatted_box['BottomX']}, BottomY={formatted_box['BottomY']}")
        
        return added_count
        
     except Exception as e:
        import traceback
        print(f"Error adding manual boxes to JSON: {str(e)}")
        print(traceback.format_exc())
        return 0
# This method is now replaced by load_manual_boxes and add_manual_boxes_to_json
# You can safely remove the old add_manual_boxes_to_wall_json method

    
if __name__ == "__main__":
    # Initialize the processor with debug mode enabled
    processor = FurnitureDetectionProcessor("/path/to/your/base/directory", debug=True)
    
    # Process an image
    result = processor.process_image("/path/to/your/image.jpg")
    
    print("Processing completed successfully.")
    print(f"Results saved to {processor.output_dir}")