import cv2 as cv
import numpy as np
from utilities import show_image, save_image, get_piece_outline, resize_image
from global_variables import colors, hardcoded_color_ranges

BOARD = cv.imread("templates/shapes/board.jpg")
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
    for color in colors:
        lower, upper = hardcoded_color_ranges["black"]
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
