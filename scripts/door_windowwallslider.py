import json
import os
import math
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Load data from file

def calculate_slope_and_angle(keypoints):
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

def process_json(data, x_scale, y_scale):
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
    
    # A2, A3, ...: Process furniture data
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
        
        slope, angle, facing, top, bottom = calculate_slope_and_angle(obj["keypoints"])
        
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
            "width": obj["width"] * x_scale,
            "height": obj["height"] * y_scale,
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
def process_furniture_data(x_scale, y_scale, threshold_height=113, threshold_width=113):
    input_file = r"V:\One knot One\test\WALLYOLO.json"
    
    if not os.path.exists(input_file):
        print("FURNITURE.json not found.")
        return
    
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    wall_data = {}
    slider_data = {}
    window_data = {}

    for i, obj in enumerate(data.values(), start=1):
        if isinstance(obj, dict) and "name" in obj:
            width = obj["width"]
            height = obj["height"]
            orientation = obj["orientation"]
            
            converted_to_slider = False
            
            if obj["name"].lower() == "window":
                if orientation == "Vertical" and height > threshold_height:
                    obj["name"] = "slider"
                    converted_to_slider = True
                elif orientation == "Horizontal" and width > threshold_width:
                    obj["name"] = "slider"
                    converted_to_slider = True
            
            if converted_to_slider:
                slider_data[f"A{i}"] = obj
            else:
                window_data[f"A{i}"] = obj
        
            # Include walls, doors, and converted sliders in wall_data
            if obj["name"].lower() in ["wall", "door", "slider"]:
                wall_data[f"A{i}"] = obj
    
    # ✅ Include non-converted windows
    
    for key, value in window_data.items():
        if value["name"].lower() == "window":
            wall_data[key] = value
    
    output_file = r"V:\One knot One\test\WALLYOLO1.json"
    with open(output_file, 'w') as file:
        json.dump(wall_data, file, indent=4)
    
    print(f"Processed wall and slider data saved to {output_file}")



# Chessboard distance (Chebyshev distance)



def main():
 script_dir = os.path.dirname(os.path.abspath(__file__))
 scale_factors_path = os.path.join(script_dir, "scale_factors.json")
    
 with open(scale_factors_path, "r") as f:
        scale_factors = json.load(f)
    
 x_scale = scale_factors.get("scale_x", 1)
 y_scale = scale_factors.get("scale_y", 1)
 file_path = r"V:\One knot One\test\WALL.json"
 output_file_path = r"V:\One knot One\test\WALLYOLO.json"

 with open(file_path, 'r') as file:
        data = json.load(file)
    
 processed_data = process_json(data, x_scale, y_scale)
    
 with open(output_file_path, 'w') as file:
        json.dump(processed_data, file, indent=4)
    
 print(f"Processed furniture data saved to {output_file_path}")
    
 process_furniture_data(x_scale, y_scale)
 file_path = r"V:\One knot One\test\WALLYOLO1.json"
 with open(file_path, "r") as file:
    data = json.load(file)
 def chessboard_distance(x1, y1, x2, y2):
    return max(abs(x2 - x1), abs(y2 - y1))

 def update_door_coordinates(data, threshold):
    for key, obj in data.items():
        if obj["name"] == "door":
            min_top_distance = float('inf')
            min_bottom_distance = float('inf')
            new_top_x, new_top_y = obj["TopX"], obj["TopY"]
            new_bottom_x, new_bottom_y = obj["BottomX"], obj["BottomY"]
            
            for target_key, target_obj in data.items():
                if target_obj["name"] in ["wall", "window", "slider"]:
                    # Compare door top coordinates with target coordinates
                    top_distance_1 = chessboard_distance(obj["TopX"], obj["TopY"], target_obj["x1"], target_obj["y1"])
                    top_distance_2 = chessboard_distance(obj["TopX"], obj["TopY"], target_obj["x2"], target_obj["y2"])

                    # Update top coordinates if a closer object within threshold is found
                    if top_distance_1 < threshold and top_distance_1 < min_top_distance:
                        min_top_distance = top_distance_1
                        new_top_x, new_top_y = target_obj["x1"], target_obj["y1"]
                    if top_distance_2 < threshold and top_distance_2 < min_top_distance:
                        min_top_distance = top_distance_2
                        new_top_x, new_top_y = target_obj["x2"], target_obj["y2"]

                    # Compare door bottom coordinates with target coordinates
                    bottom_distance_1 = chessboard_distance(obj["BottomX"], obj["BottomY"], target_obj["x1"], target_obj["y1"])
                    bottom_distance_2 = chessboard_distance(obj["BottomX"], obj["BottomY"], target_obj["x2"], target_obj["y2"])

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

# Threshold value (set as needed)
 threshold = 15

# Update the door coordinates based on nearest wall, window, or slider coordinates
 updated_data = update_door_coordinates(data, threshold)

# Save updated data to file
 output_file_path = r"V:\One knot One\test\WALLYOLO2.json"
 with open(output_file_path, "w") as file:
    json.dump(updated_data, file, indent=4)

 print(f"Updated data saved to {output_file_path}")
if __name__ == "__main__":
    main()
