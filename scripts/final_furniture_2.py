import math
import json
import tkinter as tk
from tkinter import filedialog
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

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
def select_input_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select JSON Input File", filetypes=[("JSON files", "*.json")])
    return file_path
def main():
    # Read scale factors from JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scale_factors_path = os.path.join(script_dir, "scale_factors.json")
    
    with open(scale_factors_path, "r") as f:
        scale_factors = json.load(f)
    
    x_scale = scale_factors.get("scale_x", 1)
    y_scale = scale_factors.get("scale_y", 1)

    # Select input file
    input_file_path = select_input_file()
    if not input_file_path:
        print("No file selected. Exiting.")
        return
   
    output_file_path = r"V:\One knot One\test\FURNITURE.json"

    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    processed_data = process_json(data, x_scale, y_scale)
    
    with open(output_file_path, 'w') as file:
        json.dump(processed_data, file, indent=4)
    
    print(f"Processed furniture data saved to {output_file_path}")
    

if __name__ == "__main__":
    main()
