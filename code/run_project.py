import cv2 as cv
import numpy as np
from board_extraction import extract_board
from utilities import show_image, save_image, compare_configs, get_piece_from_position, resize_image, draw_contours
from utilities import verify_game_and_move_index, draw_contours, get_piece_info
from utilities import draw_board_from_config_matrix as db
from get_board_configuration import get_initial_board_config, get_intermediary_board_config, get_config
from piece_classification import classify_piece_shape
from global_variables import SHAPES, BONUS
from global_variables import TEST_CONFIG, TEST_SCORES, BOARD_EXTRACTION_SAVES, BOARD_EXTRACTION_DISPLAYS
from global_variables import TRAIN_1_CONFIG, TRAIN_2_CONFIG, TRAIN_3_CONFIG, TRAIN_4_CONFIG, TRAIN_5_CONFIG
from global_variables import TRAIN_1_SCORES, TRAIN_2_SCORES, TRAIN_3_SCORES, TRAIN_4_SCORES, TRAIN_5_SCORES
from global_variables import HARDCODED_COLOR_RANGES
from hardcoded_template_maker import get_shape_templates_bonus


def get_board(displays, saves, game_index=1, move_index=0, data_type="train"):
    if not verify_game_and_move_index(game_index, move_index, data_type):
        return None

    for i in range(move_index, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        img_name = f"{game_index}_{num}.jpg"

        extract_board(img_name=img_name,
                      displays=displays, saves=saves,
                      data_type=data_type)


get_config(1, 1, "test")

# get_config(data_type="test", game_index=1, move_index=0)

# for i in range(1, 6):
#     get_board(displays=BOARD_EXTRACTION_DISPLAYS, saves=BOARD_EXTRACTION_SAVES,
#               game_index=i, move_index=0,
#               data_type="train")

#
# for i in range(0, 3):
#     get_board(displays=BOARD_EXTRACTION_DISPLAYS, saves=BOARD_EXTRACTION_SAVES,
#               game_index=i, move_index=0,
#               data_type="test")
