import cv2 as cv
import numpy as np
from utilities import show_image, save_image
from utilities import get_piece_outline, resize_image, get_contours, draw_contours, extend_piece
from global_variables import COLORS, HARDCODED_COLOR_RANGES

BOARD = cv.imread("templates/shapes/board.jpg")
BOARD_BONUS = cv.imread("templates/shapes/board_bonus.jpg")
COORDS_2816 = {
    "red": (387, 400, 535, 1076),
    "orange": (1285, 391, 530, 1075),
    "yellow": (2178, 387, 540, 1082),
    "green": (1270, 1639, 538, 1111),
    "blue": (2175, 1648, 545, 1095),
    "white": (375, 1650, 535, 1105)
}


def get_shape_templates():
    # 0: circle, 1: cross, 2: diamond, 3: square, 4: star4, 5: star8
    templates = [[], [], [], [], [], []]
    index_map = {
        0: 5,
        1: 2,
        2: 4,
        3: 1,
        4: 3,
        5: 0
    }
    for color in COLORS:
        lower, upper = HARDCODED_COLOR_RANGES["black"]
        x, y, w, h = COORDS_2816[color]

        smaller_board = BOARD[y:y + h, x:x + w]

        # show_image("b", smaller_board)

        hsv = cv.cvtColor(smaller_board, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        mask = ~mask

        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 3000 <= cv.contourArea(c) <= 25000]

        for i, c in enumerate(contours):
            x, y, w, h = cv.boundingRect(c)

            # piece = smaller_board[y:y + h, x:x + w]
            # show_image("p", piece)

            _, x, y, w, h = get_piece_outline(smaller_board, "null", x, y, w, h, False)

            template = mask[y:y + h, x:x + w]
            template = cv.medianBlur(template, 7)
            template = resize_image(template, 200, 200)

            idx = index_map[i // 3]
            templates[idx].append(template)

    for i in range(6):
        templates[i] = np.stack(templates[i], axis=0)
        result = np.max(templates[i], axis=0)

        result = cv.medianBlur(result, 11)

        # print(result.shape)

        save_image(result, "templates/shapes", f"{i}.jpg")


def get_shape_templates_bonus():
    templates = [[], [], [], [], [], []]
    lower, upper = HARDCODED_COLOR_RANGES["black"]

    hsv = cv.cvtColor(BOARD_BONUS, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)

    contours = get_contours(mask, cv.RETR_LIST, min_area=5000, max_area=20000, hw_diff_thresh=30)

    # 5, 5, 5, 0, 1, 1, 1, 3, 0, 0, 0, 2, 4, 4, 4, 2, 0, 5, 1, 1, 1, 1, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0, 1, 4,
    # 4, 4, 4, 5, 3,
    hardcoded_indices = {
        0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 1, 6: 1, 7: 3, 8: 0, 9: 0, 10: 0, 11: 2, 12: 4, 13: 4, 14: 4, 15: 2, 16: 0,
        17: 5, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 2, 25: 5, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0,
        32: 1, 33: 4, 34: 4, 35: 4, 36: 4, 37: 5, 38: 3
    }

    for i, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)

        xP, yP, wP, hP = extend_piece(50, x, y, w, h)
        piece, _, _, _, _ = get_piece_outline(BOARD_BONUS, "bonus-train 1_20", xP, yP, wP, hP)

        hsv_piece = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
        mask_piece = ~cv.inRange(hsv_piece, lower, upper)
        mask_piece = cv.medianBlur(mask_piece, 7)
        mask_piece = resize_image(mask_piece, 200, 200)

        templates[hardcoded_indices[i]].append(mask_piece)
        # show_image("t", mask_piece)

    for i in range(6):
        templates[i] = np.stack(templates[i], axis=0)
        result = np.max(templates[i], axis=0)
        result = cv.medianBlur(result, 11)
        save_image(result, "templates/bonus/shapes", f"{i}.jpg")
#