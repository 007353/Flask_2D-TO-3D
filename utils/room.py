import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import json
import os
from PIL import Image, ImageTk
from pathlib import Path
import math

# Global variables
SCALE_FACTORS = {"scale_x": 1.0, "scale_y": 1.0}
RESIZED_IMAGE_PATH = None
OUTPUT_DIRECTORY = str(Path.home() / "V:\\One knot One\\room")  # Default output directory

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

class FurnitureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Furniture Detection App")
        self.root.geometry("900x700")
        
        # Initialize variables
        self.original_image = None
        self.resized_image = None
        self.json_data = None
        self.points = []
        self.canvas_image = None
        self.scale_factors_calculated = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Create buttons - now only one button
        self.upload_image_btn = tk.Button(button_frame, text="Upload Image", command=self.upload_image, width=25)
        self.upload_image_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for image
        self.canvas_frame = tk.Frame(main_frame, bg="gray")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Status area
        self.status_frame = tk.Frame(main_frame)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.status_label = tk.Label(self.status_frame, text="Ready. Please upload an image.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Scale factors display
        self.scale_factor_frame = tk.Frame(main_frame)
        self.scale_factor_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.scale_x_label = tk.Label(self.scale_factor_frame, text="Scale X: 1.0", width=20)
        self.scale_x_label.pack(side=tk.LEFT, padx=10)
        
        self.scale_y_label = tk.Label(self.scale_factor_frame, text="Scale Y: 1.0", width=20)
        self.scale_y_label.pack(side=tk.LEFT, padx=10)
        
        # Step instructions
        self.instruction_label = tk.Label(self.scale_factor_frame, text="Step: Upload an image", font=("Arial", 10, "italic"))
        self.instruction_label.pack(side=tk.RIGHT, padx=10)
    
    def upload_image(self):
        """Upload and resize image to 800x600"""
        global RESIZED_IMAGE_PATH
        
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        
        try:
            # Load the original image
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("Could not read the image file")
            
            # Resize image to 800x600 with white padding to maintain aspect ratio
            self.resize_image()
            
            # Save the resized image
            unique_name = f"resized_image_{os.path.basename(file_path)}"
            resized_path = os.path.join(OUTPUT_DIRECTORY, unique_name)
            cv2.imwrite(resized_path, self.resized_image)
            RESIZED_IMAGE_PATH = resized_path
            
            # Display the resized image
            self.display_image(self.resized_image)
            
            # Update status
            self.status_label.config(text=f"Image resized and saved at: {resized_path}. Please select 4 points for scale calculation.")
            self.instruction_label.config(text="Select first point for X direction")
            
            # Reset points
            self.points = []
            self.scale_factors_calculated = False
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def resize_image(self):
        """Resize image to 800x600 maintaining aspect ratio with white background"""
        if self.original_image is None:
            return
        
        # Target dimensions
        target_width, target_height = 800, 600
        
        # Get original dimensions
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate aspect ratios
        img_aspect = img_width / img_height
        target_aspect = target_width / target_height
        
        # Create a white background
        resized = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        
        if img_aspect > target_aspect:
            # Image is wider than target: fit to width
            new_width = target_width
            new_height = int(new_width / img_aspect)
            
            # Resize the image
            temp = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Calculate offset to center the image vertically
            y_offset = (target_height - new_height) // 2
            
            # Place the resized image on the white canvas
            resized[y_offset:y_offset+new_height, 0:new_width] = temp
            
        else:
            # Image is taller than target: fit to height
            new_height = target_height
            new_width = int(new_height * img_aspect)
            
            # Resize the image
            temp = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Calculate offset to center the image horizontally
            x_offset = (target_width - new_width) // 2
            
            # Place the resized image on the white canvas
            resized[0:new_height, x_offset:x_offset+new_width] = temp
        
        self.resized_image = resized
    
    def display_image(self, img):
        """Display an image on the canvas"""
        if img is None:
            return
        
        # Convert OpenCV BGR to RGB
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_img = Image.fromarray(display_img)
        self.canvas_image = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
        
        # Redraw points if they exist
        self.draw_points()
    
    def draw_points(self):
        """Draw selected points on the canvas"""
        colors = ["red", "orange", "blue", "purple"]
        
        for i, point in enumerate(self.points):
            x, y = point
            color = colors[i % len(colors)]
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline=color)
            self.canvas.create_text(x+10, y-10, text=str(i+1), fill=color, font=("Arial", 12, "bold"))
            
            # Draw lines for X and Y directions
            if i == 1 and len(self.points) >= 2:
                # Line for X direction (points 0 and 1)
                self.canvas.create_line(self.points[0][0], self.points[0][1], 
                                         self.points[1][0], self.points[1][1], 
                                         fill="red", width=2)
            
            if i == 3 and len(self.points) >= 4:
                # Line for Y direction (points 2 and 3)
                self.canvas.create_line(self.points[2][0], self.points[2][1], 
                                         self.points[3][0], self.points[3][1], 
                                         fill="blue", width=2)
    
    def on_canvas_click(self, event):
        """Handle canvas click to select points for scale factor calculation"""
        if len(self.points) >= 4 or self.resized_image is None or self.scale_factors_calculated:
            return
        
        # Add the point
        self.points.append((event.x, event.y))
        
        # Redraw all points
        self.draw_points()
        
        # Update instructions
        if len(self.points) == 1:
            self.instruction_label.config(text="Select second point for X direction")
        elif len(self.points) == 2:
            self.instruction_label.config(text="Select first point for Y direction")
        elif len(self.points) == 3:
            self.instruction_label.config(text="Select second point for Y direction")
        elif len(self.points) == 4:
            self.instruction_label.config(text="Points selected - calculating scale factors...")
            # Automatically start scale factor calculation when 4 points are selected
            self.root.after(500, self.calculate_scale_factors)
    
    def calculate_scale_factors(self):
        """Calculate scale factors based on selected points"""
        global SCALE_FACTORS
        
        # Check if we have enough points
        if len(self.points) < 4:
            messagebox.showwarning("Warning", "Please select 4 points (2 for X direction, 2 for Y direction)")
            return
        
        # Calculate distances
        dx = abs(self.points[0][0] - self.points[1][0])
        dy = abs(self.points[2][1] - self.points[3][1])
        
        if dx == 0 or dy == 0:
            messagebox.showerror("Error", "Invalid points selection. Distances cannot be zero.")
            return
        
        # Ask for X and Y lengths (with feet and inches)
        x_feet = simpledialog.askfloat("Input", "Enter length in feet for X direction:")
        if x_feet is None:
            return
            
        x_inches = simpledialog.askfloat("Input", "Enter additional inches for X direction (0 if none):", initialvalue=0)
        if x_inches is None:
            x_inches = 0
        
        y_feet = simpledialog.askfloat("Input", "Enter length in feet for Y direction:")
        if y_feet is None:
            return
            
        y_inches = simpledialog.askfloat("Input", "Enter additional inches for Y direction (0 if none):", initialvalue=0)
        if y_inches is None:
            y_inches = 0
        
        # Convert to total feet (1 foot = 12 inches)
        x_length_feet = x_feet + (x_inches / 12)
        y_length_feet = y_feet + (y_inches / 12)
        
        # Convert feet to centimeters (1 foot = 30.48 cm)
        x_length_cm = x_length_feet * 30.48
        y_length_cm = y_length_feet * 30.48
        
        # Calculate scale factors
        scale_x = x_length_cm / dx
        scale_y = y_length_cm / dy
        
        # Update global scale factors
        SCALE_FACTORS = {"scale_x": scale_x, "scale_y": scale_y}
        
        # Update display
        self.scale_x_label.config(text=f"Scale X: {scale_x:.4f} cm/pixel")
        self.scale_y_label.config(text=f"Scale Y: {scale_y:.4f} cm/pixel")
        
        # Save scale factors to JSON
        scale_factors_path = os.path.join(OUTPUT_DIRECTORY, "scale_factors.json")
        with open(scale_factors_path, "w") as f:
            json.dump(SCALE_FACTORS, f, indent=4)
        
        # Update status
        self.status_label.config(text=f"Scale factors calculated. Now please upload a JSON file.")
        self.instruction_label.config(text="Upload JSON file")
        self.scale_factors_calculated = True
        
        messagebox.showinfo("Success", f"Scale factors calculated:\nX: {scale_x:.4f} cm/pixel\nY: {scale_y:.4f} cm/pixel\n\nNow please upload a JSON file.")
        
        # Automatically prompt for JSON file upload
        self.root.after(500, self.upload_json)
    
    def upload_json(self):
        """Upload and parse JSON file"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return
        
        try:
            # Read JSON file
            with open(file_path, 'r') as f:
                self.json_data = json.load(f)
            
            # Validate JSON format
            if 'predictions' not in self.json_data:
                raise ValueError("Invalid JSON format. 'predictions' key not found.")
            
            # Update status
            self.status_label.config(text=f"JSON file loaded: {file_path}")
            self.instruction_label.config(text="Processing and saving data...")
            
            # Automatically process and save
            self.root.after(500, self.process_and_save)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading JSON: {str(e)}")
    
    def process_and_save(self):
        """Process JSON data and save the result"""
        if self.json_data is None or not SCALE_FACTORS:
            messagebox.showerror("Error", "Missing data. Please complete all previous steps.")
            return
        
        try:
            # Convert to rooms_elemento.json format and apply scale factors
            output_data = self.convert_and_scale_json()
            
            # Save the processed JSON
            output_path = os.path.join(OUTPUT_DIRECTORY, "processed_rooms.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Update status
            self.status_label.config(text=f"Processed JSON saved at: {output_path}")
            self.instruction_label.config(text="Process complete")
            messagebox.showinfo("Success", f"Processing complete!\nOutput saved at: {output_path}")
            
            # Update upload button text to allow starting again
            self.upload_image_btn.config(text="Upload New Image")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing data: {str(e)}")
    
    def convert_and_scale_json(self):
        """Convert JSON format and apply scale factors"""
        output_data = {}
        scale_x = SCALE_FACTORS["scale_x"]
        scale_y = SCALE_FACTORS["scale_y"]
        
        for i, prediction in enumerate(self.json_data["predictions"], start=1):
            # Get corners from points if available, otherwise calculate from center and dimensions
            if "points" in prediction and len(prediction["points"]) >= 4:
                # Use the available points to find the corners that form the largest rectangle
                corners = self.find_rectangle_corners(prediction["points"])
            else:
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
            scaled_x = prediction["x"] * scale_x
            scaled_y = prediction["y"] * scale_y
            scaled_width = prediction["width"] * scale_x
            scaled_height = prediction["height"] * scale_y
            
            scaled_corners = [
                {"x": corners[0]["x"] * scale_x, "y": corners[0]["y"] * scale_y},  # top-left
                {"x": corners[1]["x"] * scale_x, "y": corners[1]["y"] * scale_y},  # top-right
                {"x": corners[2]["x"] * scale_x, "y": corners[2]["y"] * scale_y},  # bottom-right
                {"x": corners[3]["x"] * scale_x, "y": corners[3]["y"] * scale_y}   # bottom-left
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
    
    def find_rectangle_corners(self, points):
        """Find the four corners that form the largest rectangle from a set of points"""
        if len(points) < 4:
            # Not enough points to form a rectangle
            raise ValueError("Not enough points to form a rectangle")
        
        # Convert points to numpy array for easier manipulation
        np_points = np.array([[p["x"], p["y"]] for p in points])
        
        # Find convex hull and minimum area rectangle
        hull = cv2.convexHull(np_points.astype(np.float32))
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort the box points in clockwise order starting from top-left
        # Sort by y first, then by x for the top points
        center = np.mean(box, axis=0)
        sorted_points = sorted(box, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
        
        # Reformat to match expected dictionary format
        return [
            {"x": sorted_points[0][0], "y": sorted_points[0][1]},
            {"x": sorted_points[1][0], "y": sorted_points[1][1]},
            {"x": sorted_points[2][0], "y": sorted_points[2][1]},
            {"x": sorted_points[3][0], "y": sorted_points[3][1]}
        ]

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FurnitureApp(root)
    root.mainloop()