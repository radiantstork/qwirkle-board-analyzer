from board_extraction import extract_board
from utilities import show_image, save_image
from hardcoded_template_maker import get_number_templates, get_shape_templates
from classify_pieces import extract_pieces
import cv2 as cv
import numpy as np

board_displays = {
    # "original": True,
    "original": False,
    # "split": True,
    "split": False,
    # "thresh": True,
    "thresh": False,
    # "mblur": True,
    "mblur": False,
    # "gblur": True,
    "gblur": False,
    # "erode": True,
    "erode": False,
    # "dilate": True,
    "dilate": False,
    # "edges": True,
    "edges": False,
    # "corners": True,
    "corners": False,
    # "result": True
    "result": False
}
board_saves = {
    # "original": True,
    "original": False,
    "split": True,
    # "split": False,
    "thresh": True,
    # "thresh": False,
    "mblur": True,
    # "mblur": False,
    "gblur": True,
    # "gblur": False,
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

board_name = "3_12.jpg"
board = cv.imread(f"train/result/{board_name}")
extract_pieces(board, board_name)

# TRAIN DATA
# for i in range(1, 6):
#     for j in range(21):
#         aux = f"0{j}" if j < 10 else f"{j}"
#         img_name = f"{i}_{aux}.jpg"
#         print(img_name)
#
#         img = cv.imread(f"../images/train/{img_name}")
#         board = extract_board(img, img_name, "train", displays, saves)


# AUX DATA
# for i in range(8):
#     img_name = f"0{i + 1}.jpg"
#     print(img_name)
#
#     img = cv.imread(f"../images/aux/{img_name}")
#     board = extract_board(img, img_name, "aux_imgs", displays, saves)


# FAKE-TEST DATA
# for j in range(21):
#     aux = f"0{j}" if j < 10 else f"{j}"
#     img_name = f"1_{aux}.jpg"
#     print(img_name)
#
#     img = cv.imread(f"../images/test/fake_test/{img_name}")
#     board = extract_board(img, img_name, "fake_test", displays, saves)

