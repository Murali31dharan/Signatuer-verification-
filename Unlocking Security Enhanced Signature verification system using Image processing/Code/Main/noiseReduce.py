import cv2
import numpy

# Function to apply Non-Local Means Denoising
def noiseReduce(i1,i2):
    def denoise_image(image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Apply denoising to both images
    return denoise_image(i1),denoise_image(i2)
