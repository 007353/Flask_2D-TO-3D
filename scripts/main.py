import subprocess
import json
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
resize_script = os.path.join(script_dir, "resize.py")

mul_script = os.path.join(script_dir, "mulfac_1500_1200.py")
furniture_script = os.path.join(script_dir, "final_furniture_2.py")
door_window_wall_slider_script =os.path.join(script_dir, "door_windowwallslider.py") 

print("Executing resize.py...")
resize_process = subprocess.Popen(["python", resize_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
resize_output, resize_error = resize_process.communicate()

if resize_process.returncode != 0:
    print(f"❌ Error in resize.py:\n{resize_error}")
    exit(1)

print("✅ resize.py executed successfully!")


# 2. Execute mulfac_1500_1200.py and capture scale factors
print("Executing mulfac_1500_1200.py...")
mul_process = subprocess.Popen(["python", mul_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
mul_output, mul_error = mul_process.communicate()

if mul_process.returncode != 0:
    print(f"Error in mulfac_1500_1200.py:\n{mul_error}")
    exit(1)

# 3. Read scale factors from JSON file
scale_factors_path = os.path.join(script_dir, r"F:\Automation-Files\furniture_detection_app\flask_backend\scripts\scale_factors.json")
with open(scale_factors_path, "r") as f:
    scale_factors = json.load(f)
    
with open(r"F:\Automation-Files\furniture_detection_app\flask_backend\scripts\scale_factors.json", "w") as f:
    json.dump(scale_factors, f)
    f.flush()
json_path = os.path.abspath(r"F:\Automation-Files\furniture_detection_app\flask_backend\scripts\scale_factors.json")
print(f"Saving scale factors at: {json_path}")
scale_factors_path = os.path.join(script_dir, r"F:\Automation-Files\furniture_detection_app\flask_backend\scripts\scale_factors.json")
print(f"Reading scale factors from: {scale_factors_path}")
with open(scale_factors_path, "r") as f:
    scale_factors = json.load(f)
print(f"Scale Factors Read: {scale_factors}")


scale_x = scale_factors.get("scale_x", 1)
scale_y = scale_factors.get("scale_y", 1)

print(f"✅ Scale Factors: X = {scale_x}, Y = {scale_y}")




print("Executing final_furniture_2.py...for fuurniture")
furniture_process = subprocess.Popen(["python", furniture_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
furniture_output, furniture_error = furniture_process.communicate()

if furniture_process.returncode != 0:
    print(f"Error in final_furniture_2.py:\n{furniture_error}")
    exit(1)

print("✅ All processing completed successfully!")




print("Executing door_windowwallslider.py...")
door_process = subprocess.Popen(["python", door_window_wall_slider_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
door_output, door_error = door_process.communicate()

if door_process.returncode != 0:
    print(f"❌ Error in door_windowwallslider.py:\n{door_error}")
    exit(1)

print("✅ All processing completed successfully!")