import cv2
import numpy as np

# Load image using OpenCV
image_path = 'ameena.jpg'  # Replace with your actual image path
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    print("Error: Image not found or path is incorrect.")
    exit()

# 1. RESIZING
resized_image = cv2.resize(image, (300, 300))  # Resize to 300x300
cv2.imshow("Resized Image", resized_image)

# 2. FILTERING (BLURRING)
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
cv2.imshow("Blurred Image", blurred_image)

# 3. THRESHOLDING

# Convert to grayscale first
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded Image", thresholded_image)

# Wait and close
cv2.waitKey(0)
cv2.destroyAllWindows()
