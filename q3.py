import cv2

IMG_L_PATH = "Q3_Image/imL.png"
IMG_R_PATH = "Q3_Image/imR.png"




def disparityMap(self):
    imgLeft = cv2.imread(IMG_L_PATH)
    imgRight = cv2.imread(IMG_R_PATH)
    imgL_gray = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=21 * 16, blockSize=31)
    disparity = stereo.compute(imgL_gray, imgR_gray)
    disparity = cv2.normalize(
        disparity,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    disparity_show = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)
    cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("disparity", 600, 400)
    cv2.imshow("disparity", disparity_show)
    cv2.namedWindow("imgLeft", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("imgLeft", 600, 400)
    cv2.namedWindow("imgRight", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("imgRight", 600, 400)
    cv2.imshow("imgLeft", imgLeft)
    cv2.imshow("imgRight", imgRight)
    cv2.waitKey()

    cv2.destroyAllWindows()