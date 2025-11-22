import cv2 as cv
import numpy as np
from utilities import show_image, save_image, check_display_and_save as cdas


def resize_template(template):
    h, w = template.shape
    delta_h = 200 - h
    delta_w = 200 - w

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    return cv.copyMakeBorder(template, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)


def get_contours_for_specific_color(board, lower, upper, area):
    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower, upper)
    # show_image("mask", mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return mask, [c for c in contours if cv.contourArea(c) >= area]


def extract_and_store_templates(board, mask, templates, contours):
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(board.shape[1], x + w + padding)
        y_end = min(board.shape[0], y + h + padding)

        template = mask[y_start:y_end, x_start:x_end]
        # show_image("t", template)

        template = resize_template(template)
        # show_image(f"{i}", template)

        idx = i // 3
        templates[idx].append(template)


def get_shape_templates():
    # hardcore the data to extract the shapes from
    board = cv.imread("templates/shapes/board.jpg")

    # hardcore color properties and piece set positions
    hardcoded_colors_and_coordinates = (
        ((0, 0, 180), (179, 70, 255), (1750, 2938, 362, 944)),  # WHITE PIECES
        ((170, 100, 100), (180, 255, 255), (390, 1565, 374, 961)),  # RED PIECES
        ((5, 100, 100), (20, 255, 255), (380, 1545, 1343, 1923)),  # ORANGE PIECES
        ((25, 100, 100), (35, 255, 255), (374, 1544, 2310, 2905)),  # YELLOW PIECES
        ((40, 100, 100), (80, 255, 255), (1745, 2926, 1330, 1918)),  # GREEN PIECES
        ((100, 100, 100), (120, 255, 255), (1745, 2922, 2305, 2901)),  # BLUE PIECES
    )

    # 0: star8, 1: diamond, 2: star4, 3: cross, 4: square, 5: circle
    templates = [[], [], [], [], [], []]

    for piece_set_info in hardcoded_colors_and_coordinates:
        lower = piece_set_info[0]
        upper = piece_set_info[1]

        coords = piece_set_info[2]
        smaller_board = board[coords[0]:coords[1], coords[2]:coords[3]]

        mask, contours = get_contours_for_specific_color(smaller_board, lower, upper, 7000)

        extract_and_store_templates(board, mask, templates, contours)

    for i in range(6):
        templates[i] = np.stack(templates[i], axis=0)
        result = np.max(templates[i], axis=0)

        result = cv.medianBlur(result, 3)

        print(result.shape)

        save_image(result, "templates/shapes", f"{i}.jpg")


def get_number_templates():
    # hardcore the data to extract templates from
    board = cv.imread(f"templates/numbers/board.jpg")

    # hardcode the color properties to extract the number pieces
    lower = np.array([0, 120, 100])
    upper = np.array([10, 255, 255])
    mask, contours = get_contours_for_specific_color(board, lower, upper, 1000)
    show_image("mask", mask)

    # hardcore color properties for extraction (again)
    lower_red = np.array([0, 70, 70])
    upper_red = np.array([30, 255, 255])
    i = j = 0
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        padding = 10
        x_left = max(0, x - padding)
        y_top = max(0, y - padding)
        x_right = min(board.shape[1], x + w + padding)
        y_bottom = min(board.shape[0], y + h + padding)

        template = board[y_top:y_bottom, x_left:x_right]
        show_image("t", template)

        template = cv.cvtColor(template, cv.COLOR_BGR2HSV)
        template = cv.inRange(template, lower_red, upper_red)
        # show_image("t", template)

        area = cv.contourArea(contour)
        template = resize_template(template)
        show_image("t", template)
        if area > 5000:
            save_image(template, "templates/numbers/2", f"{i}.jpg")
            i += 1
        else:
            save_image(template, "templates/numbers/1", f"{j}.jpg")
            j += 1

    # hardcode data categorizing
    # 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    ones = [[3, 7, 11, 15, 18, 23, 27, 31, 35, 38],
            [4, 8, 13, 17, 19, 24, 29, 33, 37, 39],
            [0, 2, 6, 10, 14, 20, 22, 26, 30, 34],
            [1, 5, 9, 12, 16, 21, 25, 28, 32, 36]]
    twos = [[2, 6],
            [3, 7],
            [0, 4],
            [1, 5]]

    for i in range(4):
        imgs_ones = []
        for j in ones[i]:
            img = cv.imread(f"templates/numbers/1/{j}.jpg")
            imgs_ones.append(img)

        imgs_ones = np.stack(imgs_ones, axis=0)
        result = np.max(imgs_ones, axis=0)

        result = cv.medianBlur(result, 3)

        save_image(result, "templates/numbers/1/result", f"{i}.jpg")

        imgs_twos = []
        for j in twos[i]:
            img = cv.imread(f"templates/numbers/2/{j}.jpg")
            imgs_twos.append(img)

        imgs_twos = np.stack(imgs_twos, axis=0)
        result = np.max(imgs_twos, axis=0)

        result = cv.medianBlur(result, 3)

        save_image(result, "templates/numbers/2/result", f"{i}.jpg")