import cv2 as cv
import numpy as np
from utilities import show_image, save_image
from utilities import get_piece_outline, resize_image, get_contours, extend_piece
from global_variables import COLORS, HARDCODED_COLOR_RANGES, BONUS


def get_shape_templates():
    # 0: circle, 1: cross, 2: diamond, 3: square, 4: star4, 5: star8
    templates = [[], [], [], [], [], []]
    lower, upper = HARDCODED_COLOR_RANGES["black"]

    if BONUS:
        index_map = {
            0: 5, 1: 5, 2: 5, 3: 0, 4: 1, 5: 1, 6: 1, 7: 3, 8: 0, 9: 0, 10: 0, 11: 2, 12: 4, 13: 4, 14: 4, 15: 2, 16: 0,
            17: 5, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 2, 25: 5, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0,
            32: 1, 33: 4, 34: 4, 35: 4, 36: 4, 37: 5, 38: 3
        }

        hsv = cv.cvtColor(BOARD, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)

        contours = get_contours(mask, cv.RETR_LIST, min_area=5000, max_area=20000, hw_diff_thresh=30)

        for i, c in enumerate(contours):
            x, y, w, h = cv.boundingRect(c)

            xP, yP, wP, hP = extend_piece(50, x, y, w, h)
            piece, _, _, _, _ = get_piece_outline(BOARD, "bonus-train-1_20", xP, yP, wP, hP)

            hsv_piece = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
            mask_piece = ~cv.inRange(hsv_piece, lower, upper)
            mask_piece = cv.medianBlur(mask_piece, 7)
            mask_piece = resize_image(mask_piece, 200, 200)

            templates[index_map[i]].append(mask_piece)

    else:
        index_map = {
            0: 5,
            1: 2,
            2: 4,
            3: 1,
            4: 3,
            5: 0
        }

        for color in COLORS:
            x, y, w, h = COORDS_2816[color]

            smaller_board = BOARD[y:y + h, x:x + w]

            hsv = cv.cvtColor(smaller_board, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower, upper)
            mask = ~mask

            contours = get_contours(mask, cv.RETR_LIST, min_area=3000, max_area=25000)

            # bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            # cv.drawContours(bgr, contours, -1, (0, 0, 255), 12)
            # show_image("t", bgr)

            for i, c in enumerate(contours):
                x, y, w, h = cv.boundingRect(c)

                # piece = smaller_board[y:y + h, x:x + w]
                # show_image("p", piece)

                template = mask[y:y + h, x:x + w]
                template = cv.medianBlur(template, 7)
                template = resize_image(template, 200, 200)

                idx = index_map[i // 3]
                templates[idx].append(template)

    for k in range(6):
        templates[k] = np.stack(templates[k], axis=0)
        templates[k] = np.max(templates[k], axis=0)
        templates[k] = cv.medianBlur(templates[k], 11)
        save_image(templates[k], TEMPLATE_PATH, f"{k}.jpg")

    return templates


COORDS_2816 = {
    "red": (387, 400, 535, 1076),
    "orange": (1285, 391, 530, 1075),
    "yellow": (2178, 387, 540, 1082),
    "green": (1270, 1639, 538, 1111),
    "blue": (2175, 1648, 545, 1095),
    "white": (375, 1650, 535, 1105)
}
if BONUS:
    TEMPLATE_PATH = "templates/bonus"
else:
    TEMPLATE_PATH = "templates"

BOARD = cv.imread(f"{TEMPLATE_PATH}/aux.jpg")
TEMPLATES = get_shape_templates()
