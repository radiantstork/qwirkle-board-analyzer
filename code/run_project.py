import cv2 as cv
import numpy as np
from board_extraction import extract_board
from utilities import show_image, save_image, compare_configs, get_piece_from_position, resize_image, draw_contours
from get_board_configuration import get_initial_board_config, get_intermediary_board_config
from get_board_configuration import get_contours
from piece_classification import classify_piece_shape
from global_variables import shapes, hardcoded_color_ranges
from global_variables import train_120_shaped_pieces, templates
from global_variables import TEST_CONFIG, TEST_SCORES, BOARD_EXTRACTION_SAVES, BOARD_EXTRACTION_DISPLAYS
from global_variables import TRAIN_1_CONFIG, TRAIN_2_CONFIG, TRAIN_3_CONFIG, TRAIN_4_CONFIG, TRAIN_5_CONFIG
from global_variables import TRAIN_1_SCORES, TRAIN_2_SCORES, TRAIN_3_SCORES, TRAIN_4_SCORES, TRAIN_5_SCORES


CONFIG_LABELS_MAP = {
    "test": TEST_CONFIG,
    "train_1": TRAIN_1_CONFIG,
    "train_2": TRAIN_2_CONFIG,
    "train_3": TRAIN_3_CONFIG,
    "train_4": TRAIN_4_CONFIG,
    "train_5": TRAIN_5_CONFIG
}
SCORE_LABELS_MAP = {
    "test": TEST_SCORES,
    "train_1": TRAIN_1_SCORES,
    "train_2": TRAIN_2_SCORES,
    "train_3": TRAIN_3_SCORES,
    "train_4": TRAIN_4_SCORES,
    "train_5": TRAIN_5_SCORES
}


def get_board(displays, saves, data_type="train", save_dir="boards", game_index=1, move_index=0):
    if data_type == "test":
        read_path = "../images/test/fake_test"
        game_index = 1
    else:
        read_path = f"../images/{data_type}"

    save_path = f"{save_dir}/{data_type}"

    for i in range(move_index, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        img_name = f"{game_index}_{num}.jpg"
        img = cv.imread(f"{read_path}/{img_name}")

        extract_board(img, img_name, save_path, displays, saves)


def get_config(data_type="train", game_index=1, move_index=0):
    if data_type == "test":
        game_index = 1

    num = f"0{move_index}" if move_index < 10 else f"{move_index}"

    board_prev_name = f"{game_index}_{num}.jpg"
    board_prev = cv.imread(f"boards/{data_type}/result/{board_prev_name}")

    save_image(board_prev, f"detected_moves/{data_type}", f"{board_prev_name}")

    config = get_initial_board_config(board_prev, board_prev_name)
    scores = []
    for i in range(move_index + 1, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        board_current_name = f"{game_index}_{num}.jpg"
        board_current = cv.imread(f"boards/{data_type}/result/{board_current_name}")

        score = get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config,
                                              data_type)
        scores.append(score)

        board_prev = board_current
        board_prev_name = board_current_name

    if move_index == 0:
        key = "test" if data_type == "test" else f"train_{game_index}"
        config_label = CONFIG_LABELS_MAP[key]
        scores_label = SCORE_LABELS_MAP[key]

        mistakes = compare_configs(config, config_label)
        if tuple(scores) == scores_label:
            print("scores are correct")
        else:
            print("scores are wrong ")

        for row, col in mistakes:
            piece = get_piece_from_position(board_prev, board_prev_name, row, col)
            shape_index = classify_piece_shape(piece)
            print(f"({row},{col}) {shapes[shape_index]}")
            show_image("p", piece)


# for i in range(1, 6):
#     get_config(data_type="train", game_index=i, move_index=0)

get_config(data_type="test", game_index=1, move_index=0)

# for i in range(1, 6):
#     get_board(displays=BOARD_EXTRACTION_DISPLAYS, saves=BOARD_EXTRACTION_SAVES,
#               data_type="train", save_dir="boards",
#               game_index=i, move_index=0)

# get_board(displays=BOARD_EXTRACTION_DISPLAYS, saves=BOARD_EXTRACTION_SAVES,
#           data_type="test", save_dir="boards",
#           game_index=1, move_index=0)