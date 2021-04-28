import numpy as np
import cv2 
from matplotlib import pyplot as plt

# function needed for feature extracti
# Extraction and Matching
def extract_and_match(img1, img2):
    # first, we need to etract the features
    orb = cv2.ORB()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # now, match the features using opencv
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1,des2)

    # visualize the matches in matplotlib
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, flags=2)
    plt.imshow(img3),plt.show()

    

    
