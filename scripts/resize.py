import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageResizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Resizer & Saver")

        # Initialize variables
        self.original_image = None
        self.resized_image = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, pady=10)

        # Buttons
        btn_upload = tk.Button(button_frame, text="Upload Image", command=self.upload_image)
        btn_save = tk.Button(button_frame, text="Save Image", command=self.save_image)

        # Layout buttons
        btn_upload.pack(side=tk.LEFT, padx=5, pady=5)
        btn_save.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="gray")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.resize_image()  # Automatically resize and display
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def resize_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Upload an image first!")
            return

        # Fixed dimensions for resizing
        target_width, target_height = 800, 600

        # Get original image dimensions
        img_height, img_width = self.original_image.shape[:2]

        # Calculate scale factor to maintain aspect ratio
        scale = min(target_width / img_width, target_height / img_height)
        new_w, new_h = int(img_width * scale), int(img_height * scale)

        # Resize the image
        resized = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a white background canvas
        padded_img = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

        # Calculate top-left position for centering the image
        start_x = (target_width - new_w) // 2
        start_y = (target_height - new_h) // 2

        # Place resized image in the center of the white canvas
        padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        # Store resized image
        self.resized_image = padded_img
        self.display_image(self.resized_image)

    def save_image(self):
        if self.resized_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("BMP Image", "*.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.resized_image)
            messagebox.showinfo("Success", f"Image saved at:\n{file_path}")
        else:
            messagebox.showwarning("Warning", "Save operation canceled!")

    def display_image(self, img):
        if len(img.shape) == 2:  # If grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep a reference

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageResizerApp(root)
    root.mainloop()
