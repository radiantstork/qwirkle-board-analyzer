from board_extraction import extract_board
from utilities import show_image
import cv2 as cv

displays = {
    "original": True,
    # "original": False,
    "split": True,
    # "split": False,
    "thresh": True,
    # "thresh": False,
    "erode": True,
    # "erode": False,
    "dilate": True,
    # "dilate": False,
    "edges": True,
    # "edges": False,
    "corners": True,
    # "corners": False,
    "result": True
    # "result": False
}
saves = {
    # "original": True,
    "original": False,
    "split": True,
    # "split": False,
    "thresh": True,
    # "thresh": False,
    "erode": True,
    # "erode": False,
    "dilate": True,
    # "dilate": False,
    "edges": True,
    # "edges": False,
    "corners": True,
    # "corners": False,
    "result": True
    # "result": False
}

img = cv.imread("../images/train/1_15.jpg")
board = extract_board(img, "1_15.jpg", displays, saves)

# for i in range(1, 6):
#     for j in range(21):
#         aux = f"0{j}" if j < 10 else f"{j}"
#         img_name = f"{i}_{aux}.jpg"
#
#         img = cv.imread(f"../images/train/{img_name}")
#         board = extract_board(img, img_name, displays, saves)
