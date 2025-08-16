import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image in grayscale
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Make sure 'your_image.jpg' is in the current directory.")

# Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X-direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y-direction
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.axis('off')

plt.subplot(2, 3, 2), plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel - X'), plt.axis('off')

plt.subplot(2, 3, 3), plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel - Y'), plt.axis('off')

plt.subplot(2, 3, 4), plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel - Combined'), plt.axis('off')

plt.subplot(2, 3, 5), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.axis('off')

plt.subplot(2, 3, 6), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.axis('off')

plt.tight_layout()
plt.show()
