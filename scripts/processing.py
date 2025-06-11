import os
import json
import cv2
import math
import numpy as np
import uuid
import time
from pathlib import Path

class FurnitureDetectionProcessor:
    def __init__(self, base_dir,debug=False):
        # Directory structure
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.output_dir = os.path.join(base_dir, 'output')
        self.furniture_model_path = os.path.join(self.models_dir, 'furniture', 'weights1.pt')
        self.wall_door_model_path = os.path.join(self.models_dir, 'wall_door', 'weights2.pt')
        self.debug = debug
     
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize scale factors to None - will be set externally
        self.scale_factors = {
            "scale_x": None, 
            "scale_y": None
        }
        
        # Initialize models on demand to save memory
        self.furniture_model = None
        self.wall_door_model = None
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
        """Resize image to 800x600 with center crop"""
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
        resized_path = os.path.join(self.output_dir, f"resized_{unique_id}.jpg")
        cv2.imwrite(resized_path, final_resized)
        
        return resized_path
    
    def run_furniture_detection(self, image_path):
     """Run furniture detection model on the image and extract proper keypoints"""
    # Load the model
     model = self.load_furniture_model()
    
    # Run inference
     results = model(image_path)
    
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
                return None, json_data  # Return the JSON data directly
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
                            "fridge": {0: "bottom", 1: "top"},
                            "ArmChair":{0:"bottom",1:"top"}
                            # Add mappings for other furniture types as needed
                        }
                        
                        # Get the mapping for this furniture type (convert to lowercase for case-insensitive matching)
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
    
     return furniture_output_path, furniture_output
    def run_wall_door_detection(self, image_path):
     """Run wall/door detection model on the image and extract proper keypoints"""
    # Load the model
     model = self.load_wall_door_model()
    
    # Run inference
     results = model(image_path)
    
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
                return None, json_data  # Return the JSON data directly
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
    
     return wall_door_output_path, wall_door_output
    def process_image(self, image_path):
        """Main entry point for processing an image"""
        try:
            print(f"Processing image: {image_path}")
            
            # Validate scale factors
            if not self.scale_factors or self.scale_factors["scale_x"] is None or self.scale_factors["scale_y"] is None:
                raise ValueError("Scale factors must be set before processing the image")
            
            x_scale = self.scale_factors["scale_x"]
            y_scale = self.scale_factors["scale_y"]
            
            print(f"Using scale factors - X: {x_scale}, Y: {y_scale}")
            
            # 1. Run furniture detection
            start_time = time.time()
            _, furniture_data = self.run_furniture_detection(image_path)
            print(f"Furniture detection completed in {time.time() - start_time:.2f}s")
            
            # 2. Process furniture data with scaling
            start_time = time.time()
            furniture_processed = self.process_furniture_data(furniture_data, x_scale, y_scale)
            
            # Save processed furniture data
            furniture_output_path = os.path.join(self.output_dir, "FURNITURE.json")
            with open(furniture_output_path, 'w') as f:
                json.dump(furniture_processed, f, indent=4)
            print(f"Furniture data processed in {time.time() - start_time:.2f}s: {furniture_output_path}")
            
            # 3. Run wall/door detection
            start_time = time.time()
            _, wall_door_data = self.run_wall_door_detection(image_path)
            print(f"Wall/door detection completed in {time.time() - start_time:.2f}s")
            
            # 4. Process wall/door data
            start_time = time.time()
            wall_door_processed = self.process_wall_door_data(wall_door_data, x_scale, y_scale)
            
            # Save final processed data
            final_output_path = os.path.join(self.output_dir, "WALL.json")
            with open(final_output_path, 'w') as f:
                json.dump(wall_door_processed, f, indent=4)
            print(f"Wall/door data processed in {time.time() - start_time:.2f}s: {final_output_path}")
            
            # 5. Combine results for the frontend
            combined_results = {
                "furniture": furniture_processed,
                "wall_door": wall_door_processed,
                "image_path": image_path
            }
            
            return combined_results
            
        except Exception as e:
            import traceback
            print(f"Error processing image: {str(e)}")
            print(traceback.format_exc())
            raise
    
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
    
    def process_wall_door_data(self, data, x_scale, y_scale, threshold_height=113, threshold_width=113):
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
    
if __name__ == "__main__":
    # Initialize the processor with debug mode enabled
    processor = FurnitureDetectionProcessor("/path/to/your/base/directory", debug=True)
    
    # Process an image
    result = processor.process_image("/path/to/your/image.jpg")
    
    print("Processing completed successfully.")
    print(f"Results saved to {processor.output_dir}")