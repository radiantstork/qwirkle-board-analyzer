import cv2 as cv
from utilities import resize_image
from global_variables import colors, hardcoded_color_ranges, templates


def classify_piece_color(piece):
    # show_image("piece", piece)

    max_count = 0
    piece_color = None
    for color in colors:
        lower, upper = hardcoded_color_ranges.get(color)

        hsv = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)

        # show_image(f"{color}", mask)

        count = cv.countNonZero(mask)
        if count > max_count:
            max_count = count
            piece_color = color

    return piece_color


def classify_piece_shape(piece):
    lower, upper = hardcoded_color_ranges["black"]

    hsv = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = ~mask

    mask = resize_image(mask, 200, 200)

    best_score = -1
    shape_index = None
    for i, template in enumerate(templates):
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        result = cv.matchTemplate(mask, template, cv.TM_CCOEFF_NORMED)

        _, score, _, _ = cv.minMaxLoc(result)

        if score > best_score:
            best_score = score
            shape_index = i

    return shape_index
