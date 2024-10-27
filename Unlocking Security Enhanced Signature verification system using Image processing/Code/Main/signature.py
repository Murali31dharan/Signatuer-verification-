import cv2
from skimage.metrics import structural_similarity as ssim
from chidis import compare_histograms

# TODO add contour detection for enhanced accuracy



def match(path1, path2):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    n_img1=cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    n_img2=cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)
    # turn images to grayscale
    img1 = cv2.cvtColor(n_img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(n_img2, cv2.COLOR_BGR2GRAY)
    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    
    # display both images
    cv2.imshow("One", img1)
    cv2.imshow("Two", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     if(compare_histograms(path1,path2)<1 or)
    similarity_value = "{:.2f}".format(ssim(img1, img2)*100)
    print("Similarity Score is ", float(similarity_value),
          "type=", type(similarity_value))
    return float(similarity_value)
