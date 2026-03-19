import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract as tess
import tkinter as tk
from tkinter import filedialog, messagebox, Label

# Set path to tesseract executable (Windows)
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------- FUNCTIONS ---------------------------- #

def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    return not ((area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6))

def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        contour_area = [cv2.contourArea(c) for c in contours]
        max_cntr_index = np.argmax(contour_area)
        max_cnt = contours[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(contour_area[max_cntr_index], w, h):
            return plate, None

        final_img = thresh[y:y + h, x:x + w]
        return final_img, [x, y, w, h]
    else:
        return plate, None

def isMaxWhite(plate):
    return np.mean(plate) >= 115

def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect
    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15 or height == 0 or width == 0:
        return False

    area = height * width
    return ratioCheck(area, width, height)

def browse_image():
    global img_path
    img_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if img_path:
        process_image(img_path)
    else:
        messagebox.showwarning("Warning", "Please select an image!")

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Unable to load image!")
        return

    # Preprocessing
    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological transformation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph_img_threshold = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, element)

    # Find contours
    contours, _ = cv2.findContours(morph_img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    plate_found = False
    extracted_text = ""

    for cnt in contours:
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]

            if isMaxWhite(plate_img):
                clean_plate, rect = clean2_plate(plate_img)
                if rect:
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    plate_im = Image.fromarray(clean_plate)
                    extracted_text = tess.image_to_string(plate_im, lang='eng').strip()
                    plate_found = True

                    # Show detected number plate
                    show_plate(plate_img, extracted_text)
                    break

    if not plate_found:
        messagebox.showinfo("Result", "No number plate detected!")

def show_plate(plate_img, extracted_text):
    # Convert OpenCV image to PIL format
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    plate_pil = Image.fromarray(plate_rgb)
    plate_tk = ImageTk.PhotoImage(plate_pil)

    # Update GUI with detected plate image
    plate_label.config(image=plate_tk)
    plate_label.image = plate_tk

    # Show detected text
    result_label.config(text=f"Detected Number Plate: {extracted_text}")

# ---------------------------- GUI ---------------------------- #

root = tk.Tk()
root.title("Number Plate Recognition")
root.geometry("800x600")
root.config(bg="#222222")

# Title
title_label = tk.Label(root, text="Vehicle Number Plate Recognition", font=("Arial", 20, "bold"), fg="white", bg="#222222")
title_label.pack(pady=20)

# Browse Button
browse_button = tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=5)
browse_button.pack(pady=10)

# Label to show detected plate image
plate_label = Label(root, bg="#222222")
plate_label.pack(pady=20)

# Label to show result text
result_label = tk.Label(root, text="", font=("Arial", 16), fg="yellow", bg="#222222")
result_label.pack(pady=20)

# Exit Button
exit_button = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 14), bg="#FF3B3F", fg="white", padx=20, pady=5)
exit_button.pack(pady=10)

root.mainloop()
