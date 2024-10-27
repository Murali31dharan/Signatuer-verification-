import cv2
import numpy as np
def calculate_pressure(signature_image):
        gray_image = cv2.cvtColor(signature_image, cv2.COLOR_BGR2RGB)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        non_zero_pixels = np.count_nonzero(binary_image)
        pressure = non_zero_pixels / (binary_image.shape[0] * binary_image.shape[1])
        return pressure