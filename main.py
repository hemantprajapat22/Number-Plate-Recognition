import cv2
import numpy as np
from PIL import Image
import pytesseract as tess
from tkinter import filedialog, Tk

# Set path to tesseract executable (Windows)
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to select image using file dialog
def select_image():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path

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

# ------------------- Main Program -------------------

# Select image
img_path = select_image()
if not img_path:
    print("No image selected. Exiting...")
    exit()

# Load image
img = cv2.imread(img_path)
if img is None:
    print("Error: Unable to load image!")
    exit()

print("Processing input image...")
cv2.imshow("Input", img)

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
print("Total contours found:", len(contours))
cv2.drawContours(img2, contours, -1, (0, 255, 0), 1)

plate_found = False

for cnt in contours:
    min_rect = cv2.minAreaRect(cnt)
    if ratio_and_rotation(min_rect):
        x, y, w, h = cv2.boundingRect(cnt)
        plate_img = img[y:y + h, x:x + w]
        print("Potential number plate detected...")
        cv2.imshow("Number Plate", plate_img)

        if isMaxWhite(plate_img):
            clean_plate, rect = clean2_plate(plate_img)
            if rect:
                x1, y1, w1, h1 = rect
                x, y, w, h = x + x1, y + y1, w1, h1
                plate_im = Image.fromarray(clean_plate)
                text = tess.image_to_string(plate_im, lang='eng')
                print("Detected Number Plate Text:", text.strip())
                plate_found = True
                break

if not plate_found:
    print("No number plate detected!")

cv2.waitKey(0)
cv2.destroyAllWindows()
