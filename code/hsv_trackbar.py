import cv2
import numpy as np

# --- Configuration ---
img_path = "board_extraction/train/result/5_00.jpg"


# ---------------------

def nothing(x):
    """Placeholder function for trackbar creation."""
    pass


# Load the image
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

# Resize image (Optional, but good practice for large images)
# img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

# Convert the image to HSV color space once
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Use one main window name for both the trackbars and the primary output
TRACKBAR_WINDOW = 'HSV Color Adjuster'
cv2.namedWindow(TRACKBAR_WINDOW)

# Create trackbars (Sliders)
cv2.createTrackbar('H Lower', TRACKBAR_WINDOW, 0, 179, nothing)
cv2.createTrackbar('H Higher', TRACKBAR_WINDOW, 179, 179, nothing)
cv2.createTrackbar('S Lower', TRACKBAR_WINDOW, 0, 255, nothing)
cv2.createTrackbar('S Higher', TRACKBAR_WINDOW, 255, 255, nothing)
cv2.createTrackbar('V Lower', TRACKBAR_WINDOW, 0, 255, nothing)
cv2.createTrackbar('V Higher', TRACKBAR_WINDOW, 255, 255, nothing)

# Set initial trackbar positions for white
cv2.setTrackbarPos('H Lower', TRACKBAR_WINDOW, 0)
cv2.setTrackbarPos('H Higher', TRACKBAR_WINDOW, 179)
cv2.setTrackbarPos('S Lower', TRACKBAR_WINDOW, 0)
cv2.setTrackbarPos('S Higher', TRACKBAR_WINDOW, 30)
cv2.setTrackbarPos('V Lower', TRACKBAR_WINDOW, 220)
cv2.setTrackbarPos('V Higher', TRACKBAR_WINDOW, 255)

while True:
    # 1. Get current positions of the trackbars
    hL = cv2.getTrackbarPos('H Lower', TRACKBAR_WINDOW)
    hH = cv2.getTrackbarPos('H Higher', TRACKBAR_WINDOW)
    sL = cv2.getTrackbarPos('S Lower', TRACKBAR_WINDOW)
    sH = cv2.getTrackbarPos('S Higher', TRACKBAR_WINDOW)
    vL = cv2.getTrackbarPos('V Lower', TRACKBAR_WINDOW)
    vH = cv2.getTrackbarPos('V Higher', TRACKBAR_WINDOW)

    # 2. Define the Lower and Upper HSV bounds
    LowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    # 3. Threshold the HSV image
    mask = cv2.inRange(hsv, LowerRegion, upperRegion)

    # 4. Display the results
    # Use the trackbar window to display the mask
    cv2.imshow(TRACKBAR_WINDOW, mask)
    cv2.imshow("Original Image", img)  # Keep this separate for reference

    # 5. Wait for key press
    # A wait time of 1ms is usually fine, but 10ms can sometimes be more stable.
    # We use cv2.waitKey() to refresh the window and check for keyboard input.
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Lower HSV: ({hL}, {sL}, {vL})")
print(f"Upper HSV: ({hH}, {sH}, {vH})")