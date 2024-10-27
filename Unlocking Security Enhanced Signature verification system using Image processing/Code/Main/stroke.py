import cv2
import numpy as np
from noiseReduce import noiseReduce

def stroke_rate(i1,i2):
    def calculate_stroke(signature_image):
        gray_image = cv2.cvtColor(signature_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return len(largest_contour)
    return "Stroke for Image 1:", calculate_stroke(i1),"Stroke for Image 2:", calculate_stroke(i2)

