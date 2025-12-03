import cv2 as cv
from utilities import resize_image, show_image
from global_variables import COLORS, HARDCODED_COLOR_RANGES, TEMPLATES


def classify_piece_color(piece):
    max_count = 0
    piece_color = None
    for color in COLORS:
        lower, upper = HARDCODED_COLOR_RANGES.get(color)

        hsv = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)

        count = cv.countNonZero(mask)
        if count > max_count:
            max_count = count
            piece_color = color

    return piece_color


def classify_piece_shape(piece):
    lower, upper = HARDCODED_COLOR_RANGES["black"]

    hsv = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = ~mask

    mask = resize_image(mask, 200, 200)

    mask = cv.medianBlur(mask, 3)

    best_score = -1
    shape_index = None
    for i, template in enumerate(TEMPLATES):
        result = cv.matchTemplate(mask, template, cv.TM_CCOEFF_NORMED)

        _, score, _, _ = cv.minMaxLoc(result)
        if score > best_score:
            best_score = score
            shape_index = i

    return shape_index
