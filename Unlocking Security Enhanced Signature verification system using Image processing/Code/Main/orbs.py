
import cv2
import matplotlib.pyplot as plt

def orb(i1,i2):
    def detect_orb_features(image_path):
        # Convert image to grayscale
        gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
        # Initialize ORB detector
        orb = cv2.ORB_create()
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image_path, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints, keypoints, descriptors
    # Detect ORB features for the first image
    image1_with_keypoints, keypoints1, descriptors1 = detect_orb_features(i1)
    # Detect ORB features for the second image
    image2_with_keypoints, keypoints2, descriptors2 = detect_orb_features(i2)
    # Plot the images with ORB keypoints
    plt.figure(figsize=(10, 5))
    # Plot image 1
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoints - Image 1')
    plt.axis('off')
    # Plot image 2
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoints - Image 2')
    plt.axis('off')
    plt.show()
