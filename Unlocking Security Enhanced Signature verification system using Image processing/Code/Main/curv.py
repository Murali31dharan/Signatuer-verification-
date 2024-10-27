import cv2
import numpy as np
import matplotlib.pyplot as plt

def Curv(i1,i2):
    s1=i1
    s2=i2
    def calculate_curvature(signature_image):
        # Convert signature image to grayscale
        gray = cv2.cvtColor(signature_image, cv2.COLOR_BGR2GRAY)
        # Threshold the image to obtain a binary image
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Compute curvature for each contour
        curvatures = []
        for contour in contours:
            # Fit a circle to each segment of the contour
            for i in range(len(contour)):
                # Extract a small segment of the contour
                segment = contour[i:i+10]  # You can adjust the segment size as needed
                # Skip if the segment is too small
                if len(segment) < 3:
                    continue
                # Fit a circle to the segment
                (x, y), radius = cv2.minEnclosingCircle(segment)
                # Calculate curvature as the reciprocal of the radius
                curvature = 1 / radius if radius > 0 else 0
                curvatures.append(curvature)
        return curvatures
    # Calculate curvature for each signature
    curvature1 = calculate_curvature(s1)
    curvature2 = calculate_curvature(s2)
    # Plot curvature distributions
    plt.figure(figsize=(10, 5))
    plt.hist(curvature1, bins=20, alpha=0.5, color='blue', label='Signature 1')
    plt.hist(curvature2, bins=20, alpha=0.5, color='green', label='Signature 2')
    plt.xlabel('Curvature')
    plt.ylabel('Frequency')
    plt.title('Curvature Analysis')
    plt.legend()