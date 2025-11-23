from board_extraction import extract_board
from utilities import show_image, save_image, draw_board_from_config_matrix as db, compare_configs
from utilities import get_piece_from_position
from hardcoded_template_maker import get_number_templates, get_shape_templates
from get_board_configuration import get_intermediary_board_config, get_initial_board_config
from global_variables import train_1_scores, train_2_scores, train_3_scores, train_4_scores, train_5_scores
from global_variables import train_1_config, train_2_config, train_3_config, train_4_config, train_5_config
from global_variables import train_120_shaped_pieces, shapes
from global_variables import test_scores, test_config
from piece_classification import classify_piece_shape
import cv2 as cv

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

board_prev_name = "1_00.jpg"
board_prev = cv.imread(f"board_extraction/test/result/{board_prev_name}")
save_image(board_prev, "detected_moves/test/moves", f"{board_prev_name}")
config = get_initial_board_config(board_prev, board_prev_name)
scores = []
for i in range(1, 21):
    num = f"0{i}" if i < 10 else f"{i}"

    board_current_name = f"1_{num}.jpg"
    board_current = cv.imread(f"board_extraction/test/result/{board_current_name}")

    score = get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config)
    scores.append(score)

    board_prev = board_current
    board_prev_name = board_current_name

mistakes = compare_configs(config, test_config)
print(tuple(scores) == test_scores)

for row, col in mistakes:
    piece = get_piece_from_position(board_prev, board_prev_name, row, col)
    shape_index = classify_piece_shape(piece)
    print(f"({row},{col}) {shapes[shape_index]}")
    show_image("p", piece)

# img = db(config)
# for line in config:
#     print("[", end="")
#     for x in line:
#         print(f"\"{x}\", ", end="")
#     print()
# show_image(f"test", img)
# save_image(img, "detected_moves/test/result", f"test.jpg")

# for i in range(1, 6):
# board_prev_name = f"{i}_00.jpg"
# board_prev = cv.imread(f"board_extraction/train/result/{board_prev_name}")

# config = get_initial_board_config(board_prev, board_prev_name)

# for j in range(1, 21):
#     num = f"0{j}" if j < 10 else f"{j}"
#
# board_current_name = f"{i}_{num}.jpg"
# board_current = cv.imread(f"board_extraction/train/result/{board_current_name}")
#
#     get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config)
#
#     board_prev = board_current
#     board_prev_name = board_current_name

# print_pieces(board_prev, config)

# img = db(config)
# show_image(f"game {i}", img)
# save_image(img, "detected_moves/train/result", f"game {i}.jpg")

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
