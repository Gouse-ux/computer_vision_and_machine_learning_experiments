import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')  # Replace with your actual image path
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------ HOG Feature Extraction ------------------
# Create HOG descriptor
hog = cv2.HOGDescriptor()
hog_features = hog.compute(gray)

print(f"HOG feature vector shape: {hog_features.shape}")

# ------------------ SIFT Feature Extraction ------------------
# Create SIFT object
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
sift_img = cv2.drawKeypoints(image, keypoints_sift, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ------------------ ORB Feature Extraction ------------------
# Create ORB detector
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
orb_img = cv2.drawKeypoints(image, keypoints_orb, None, color=(0, 255, 0), flags=0)

# ------------------ Display Outputs ------------------
cv2.imshow('Original Image', image)
cv2.imshow('SIFT Features', sift_img)
cv2.imshow('ORB Features', orb_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
