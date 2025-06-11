import tkinter as tk
from tkinter import simpledialog, filedialog,messagebox
import cv2
import numpy as np
import json

global points, img
points = []
TARGET_WIDTH = 800
TARGET_HEIGHT = 600

def convert_length(feet):
    return feet * 30.48  # Convert feet to cm

def manhattan_distance(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return dx, dy

def get_length(direction):
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askfloat("Input", f"Enter length in feet for {direction} direction:")
    return convert_length(user_input) if user_input else None

def process_x_direction():
    global points,scale_x
    print(f"Selected Points for X: {points[0]}, {points[1]}")
    dx, _ = manhattan_distance(points[0], points[1])
    print(f"Manhattan Distance X = {dx} pixels")
    
    length_cm_x = get_length("X")
    if length_cm_x is None:
        messagebox.showerror("Error", "Invalid input for length in X direction")
        return
    
    scale_x = length_cm_x / dx if dx != 0 else 0
    print(f"Scale Factor X = {scale_x:.4f} cm/pixel")
    messagebox.showinfo("Result", f"Scale X: {scale_x:.4f} cm/pixel\nNow select two points for Y direction.")

def process_y_direction():
    global points,scale_y
    print(f"Selected Points for Y: {points[2]}, {points[3]}")
    _, dy = manhattan_distance(points[2], points[3])
    print(f"Manhattan Distance Y = {dy} pixels")
    
    length_cm_y = get_length("Y")
    if length_cm_y is None:
        messagebox.showerror("Error", "Invalid input for length in Y direction")
        return
    
    scale_y = length_cm_y / dy if dy != 0 else 0
    print(f"Scale Factor Y = {scale_y:.4f} cm/pixel")
    messagebox.showinfo("Result", f"Scale Y: {scale_y:.4f} cm/pixel\nCalculation Complete.")

def upload_image():
    global img, points
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        original_img = cv2.imread(file_path)
        img = cv2.resize(original_img, (TARGET_WIDTH, TARGET_HEIGHT))
        cv2.imshow("Select Points", img)
        cv2.setMouseCallback("Select Points", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", img)
        if len(points) == 2:
            process_x_direction()
        elif len(points) == 4:
            process_y_direction()
            # Save scale factors to JSON
            scale_factors = {"scale_x": scale_x, "scale_y": scale_y}
            with open(r"F:\Automation-Files\furniture_detection_app\flask_backend\scripts\scale_factors.json", "w") as f:
                json.dump(scale_factors, f)
            print(json.dumps(scale_factors))  # Print output for main.py to capture
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    upload_image()
