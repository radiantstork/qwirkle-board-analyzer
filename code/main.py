from board_extraction import extract_board
from utilities import show_image, save_image, draw_board_from_config_matrix as db
from hardcoded_template_maker import get_number_templates, get_shape_templates
from get_board_configuration import get_intermediary_board_config, get_initial_board_config
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

# board_prev_name = "1_00.jpg"
# board_prev = cv.imread(f"board_extraction/train/result/{board_prev_name}")
# board_current_name = "1_01.jpg"
# board_current = cv.imread(f"board_extraction/train/result/{board_current_name}")
# config = get_initial_board_config(board_prev)
# config = get_intermediary_board_config(board_current, board_prev, config)

# board_prev_name = "1_00.jpg"
# board_prev = cv.imread(f"board_extraction/train/result/{board_prev_name}")
# config = get_initial_board_config(board_prev)
# img = db(config)
# show_image("c", img)
# for j in range(21):
#     num = f"0{j}" if j < 10 else f"{j}"
#
#     board_current_name = f"1_{num}.jpg"
#     board_current = cv.imread(f"board_extraction/test/result/{board_current_name}")
#
#     get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config)
#
#     board_prev = board_current
#     board_prev_name = board_current_name
# img = db(config)
# show_image("test", img)
# save_image(img, "detected_moves/test/result", "1.jpg")


for i in range(4, 5):
    board_prev_name = f"{i}_00.jpg"
    board_prev = cv.imread(f"board_extraction/train/result/{board_prev_name}")

    config = get_initial_board_config(board_prev)

    # for j in range(1, 21):
    #     num = f"0{j}" if j < 10 else f"{j}"
    #
    #     board_current_name = f"{i}_{num}.jpg"
    #     board_current = cv.imread(f"board_extraction/train/result/{board_current_name}")
    #
    #     get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config)
    #
    #     board_prev = board_current
    #     board_prev_name = board_current_name

    img = db(config)
    show_image(f"game {i}", img)
    save_image(img, "detected_moves/train/result", f"game {i}.jpg")

# for i in range(1, 6):
#     board_last_name = f"{i}_00.jpg"
#     board_last = cv.imread(f"train/result/{board_last_name}")


#     config = get_initial_board_config(board_last, board_last_name)
#
#     for j in range(1, 21):
#         num = f"0{j}" if j < 10 else f"{j}"
#
#         board_current_name = f"{i}_{num}.jpg"
#         board_current = cv.imread(f"train/result/{board_current_name}")
#
#         config = get_board_config_from_last_config(board_current, board_current_name, board_last, board_last_name, config)
#
#         board_last_name = board_current_name
#         board_last = board_current
#
#     db(config, f"game {i}")

# board_last_name = f"1_00.jpg"
# board_last = cv.imread(f"fake_test/result/{board_last_name}")
# config = get_initial_board_config(board_last, board_last_name)
# for j in range(1, 21):
#     num = f"0{j}" if j < 10 else f"{j}"
#
#     board_current_name = f"1_{num}.jpg"
#     board_current = cv.imread(f"fake_test/result/{board_current_name}")
#
#     config = get_board_config_from_last_config(board_current, board_current_name, board_last, board_last_name, config)
#
#     board_last_name = board_current_name
#     board_last = board_current
#
# db(config, "fake test")
