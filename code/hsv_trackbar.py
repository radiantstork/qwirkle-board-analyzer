import cv2 as cv
import numpy as np


def nothing(x):
    pass


img_path = "../images/train/bonus/1_20.jpg"
img = cv.imread(img_path)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

TRACKBAR_WINDOW = "color adjuster"
cv.namedWindow(TRACKBAR_WINDOW)

cv.createTrackbar("H lower", TRACKBAR_WINDOW, 0, 179, nothing)
cv.createTrackbar("H higher", TRACKBAR_WINDOW, 179, 179, nothing)
cv.createTrackbar("S lower", TRACKBAR_WINDOW, 0, 255, nothing)
cv.createTrackbar("S higher", TRACKBAR_WINDOW, 255, 255, nothing)
cv.createTrackbar("V lower", TRACKBAR_WINDOW, 0, 255, nothing)
cv.createTrackbar("V higher", TRACKBAR_WINDOW, 255, 255, nothing)

cv.setTrackbarPos("H lower", TRACKBAR_WINDOW, 0)
cv.setTrackbarPos("H higher", TRACKBAR_WINDOW, 255)
cv.setTrackbarPos("S lower", TRACKBAR_WINDOW, 0)
cv.setTrackbarPos("S higher", TRACKBAR_WINDOW, 255)
cv.setTrackbarPos("V lower", TRACKBAR_WINDOW, 0)
cv.setTrackbarPos("V higher", TRACKBAR_WINDOW, 255)

while True:
    hL = cv.getTrackbarPos("H lower", TRACKBAR_WINDOW)
    hH = cv.getTrackbarPos("H higher", TRACKBAR_WINDOW)
    sL = cv.getTrackbarPos("S lower", TRACKBAR_WINDOW)
    sH = cv.getTrackbarPos("S higher", TRACKBAR_WINDOW)
    vL = cv.getTrackbarPos("V lower", TRACKBAR_WINDOW)
    vH = cv.getTrackbarPos("V higher", TRACKBAR_WINDOW)

    lowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    mask = cv.inRange(hsv, lowerRegion, upperRegion)

    cv.imshow(TRACKBAR_WINDOW, mask)
    cv.imshow("original", img)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv.destroyAllWindows()
print(f"lower HSV: ({hL}, {sL}, {vL})")
print(f"upper HSV: ({hH}, {sH}, {vH})")
