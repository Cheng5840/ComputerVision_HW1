import cv2
import numpy as np


IMG_L_PATH = "Q4_Image/Left.jpg"
IMG_R_PATH = "Q4_Image/Right.jpg"


def createKeyPoint(self):
    img1 = cv2.imread(IMG_L_PATH)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift= cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img1_gray, None)

    kp_image1 = cv2.drawKeypoints(img1_gray, keypoints, None, color=(0, 255, 0))
    kp_image1 = cv2.resize(kp_image1, (800, 600))
    cv2.imshow("Keypoints", kp_image1)
    cv2.resizeWindow("Keypoints", 800, 600)
    cv2.waitKey()
    cv2.destroyAllWindows()


def matchedKeyPoint(self):
    img1 = cv2.imread(IMG_L_PATH)
    img2 = cv2.imread(IMG_R_PATH)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    SIFT = cv2.SIFT_create()
    key1, des1 = SIFT.detectAndCompute(img1_gray, None)
    key2, des2 = SIFT.detectAndCompute(img2_gray, None)

    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)

    goodMatches = []
    minRatio = 0.75
    for m, n in matches:
        if m.distance < minRatio * n.distance:
            goodMatches.append(m)

  
    img = cv2.drawMatches(img1, key1, img2, key2, goodMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img = cv2.resize(img, (1300, 800))
    cv2.imshow("Matched", img)
    cv2.resizeWindow("Matched", 1300, 800)
    cv2.waitKey()
    cv2.destroyAllWindows()