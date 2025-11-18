from board_extraction import extract_board
from utilities import show_image, save_image
from template_maker import extract_shapes, extract_numbers
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

template_displays = {}
template_saves = {
    "original": True,
    "split": True,
    "thresh": True,
    "dilate": True,
    "erode": True,
    "union": True
}

# SPECIFIC IMAGE
# img_name = "1_00.jpg"
# img = cv.imread(f"../ima

extract_numbers()
# ones = [[3, 7, 11, 15, 18, 23, 27, 31, 35, 38],
#         [4, 8, 13, 17, 19, 24, 29, 33, 37, 39],
#         [0, 2, 6, 10, 14, 20, 22, 26, 30, 34],
#         [1, 5, 9, 12, 16, 21, 25, 28, 32, 36]]
# twos = [[2, 6],
#         [3, 7],
#         [0, 4],
#         [1, 5]]
#
# for i in range(4):
#     imgs_ones = []
#     for j in ones[i]:
#         img = cv.imread(f"templates/numbers/1/{j}.jpg")
#         imgs_ones.append(img)
#
#     imgs_ones = np.stack(imgs_ones, axis=0)
#     result = np.max(imgs_ones, axis=0)
#
#     save_image(result, "templates/numbers/1/result", f"{i}.jpg")
#
#     imgs_twos = []
#     for j in twos[i]:
#         img = cv.imread(f"templates/numbers/2/{j}.jpg")
#         imgs_twos.append(img)
#
#     imgs_twos = np.stack(imgs_twos, axis=0)
#     result = np.max(imgs_twos, axis=0)
#
#     result = cv.medianBlur(result, 3)
#
#     save_image(result, "templates/numbers/2/result", f"{i}.jpg")


# board_name = "3_12.jpg"
# board = cv.imread(f"train/result/{board_name}")
# extract_pieces(board, board_name)

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

