import cv2 as cv
import numpy as np
from utilities import show_image, save_image, load_templates, draw_board_from_config_matrix as db, draw_contours

hardcoded_color_ranges = {
    "red": ((170, 100, 100), (180, 255, 255)),
    "orange": ((6, 46, 97), (20, 255, 255)),
    "yellow": ((25, 100, 100), (35, 255, 255)),
    "green": ((59, 174, 0), (80, 255, 255)),
    "blue": ((100, 100, 100), (120, 255, 255)),
    "white": ((30, 0, 212), (151, 31, 255)),
    "black": ((0, 0, 0), (179, 255, 85))
}
colors = ("red", "orange", "yellow", "green", "blue", "white")
shapes = ("star8", "diamond", "star4", "cross", "square", "circle")
mask_cache = {}
templates = load_templates()


def get_mask(board):
    lower, upper = hardcoded_color_ranges["black"]
    lower = np.array(lower)
    upper = np.array(upper)

    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    # show_image("original mask", mask)

    mask = ~mask
    # show_image("negated mask", mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=4)
    # show_image("eroded mask", mask)

    mask = ~mask
    # show_image("negated mask x2", mask)

    # save_image(mask, "../code", "mask_black.jpg")

    return mask


def get_contours(mask, list_type, min_area=0, max_area=float("inf"), hw_diff_thresh=15):
    if list_type == "external":
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    elif list_type == "list":
        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    else:
        return None

    filtered_contours = []
    for c in contours:
        if min_area <= cv.contourArea(c) <= max_area:
            _, _, w, h = cv.boundingRect(c)
            if abs(w - h) <= hw_diff_thresh:
                filtered_contours.append(c)

    return filtered_contours


def get_cell_index_for_contour(board, contour):
    x, y, w, h = cv.boundingRect(contour)

    area_of_interest = board[y:y + h, x:x + w]

    x_center = x + w // 2
    y_center = y + h // 2

    col = x_center // 192
    row = y_center // 192

    return row, col, area_of_interest


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


def get_mask_diff(board_current, board_prev, board_current_name, board_prev_name):
    kernel = np.ones((5, 5), np.uint8)

    board_current = cv.GaussianBlur(board_current, (55, 55), 0)
    board_prev = cv.GaussianBlur(board_prev, (55, 55), 0)
    # show_image("p", board_prev)
    # show_image("c", board_current)

    mask_current = get_mask(board_current)
    mask_cache[board_current_name[:-4]] = mask_current
    if board_prev_name[:-4] in mask_cache:
        mask_prev = mask_cache[board_prev_name[:-4]]
    else:
        mask_prev = get_mask(board_prev)
    # show_image("mask prev", mask_prev)
    # show_image("mask current", mask_current)

    mask_diff = np.subtract(mask_current, mask_prev)
    # show_image("og", mask_diff)
    # save_image(mask_diff, "detected_moves/train/diff", board_current_name)

    mask_diff = cv.medianBlur(mask_diff, 13)
    # show_image("mblur", mask_diff)
    # save_image(mask_diff, "detected_moves/train/mblur1", board_current_name)

    mask_diff = cv.erode(mask_diff, kernel, iterations=2)
    # show_image("erode", mask_diff)
    # save_image(mask_diff, "detected_moves/train/erode", board_current_name)

    mask_diff = cv.medianBlur(mask_diff, 13)
    # show_image("mblur", mask_diff)
    # save_image(mask_diff, "detected_moves/train/mblur2", board_current_name)

    # mask_diff = cv.erode(mask_diff, kernel, iterations=1)
    # # show_image("erode", mask_diff)
    # save_image(mask_diff, "detected_moves/train/erode2", board_current_name)

    mask_diff = cv.dilate(mask_diff, kernel, iterations=12)
    # show_image("dilate", mask_diff)
    # save_image(mask_diff, "detected_moves/train/dilate", board_current_name)

    return mask_diff


def get_pieces_from_contour(board, board_name, contour, config):
    crop_amount = 30
    x, y, w, h = cv.boundingRect(contour)
    x += crop_amount
    y += crop_amount
    w -= crop_amount
    h -= crop_amount

    row_count = round(h / 192)
    column_count = round(w / 192)

    if row_count == 0 or column_count == 0:
        print(f"0 rows, 0 columns {board_name}")
        return None
    if row_count > 1 and column_count > 1:
        print(f"2+ rows 2+ columns {board_name}")
        return None

    w_piece = w // column_count
    h_piece = h // row_count

    crop_amount = 30

    if row_count == 1 and column_count == 1:
        x_center = x + w // 2
        y_center = y + h // 2

        row = y_center // 192
        col = x_center // 192

        if config[row][col] != "0":
            if config[row][col] == "1":
                print("piece placed on 1")
            elif config[row][col] == "2":
                print("piece placed on 2")
            else:
                print(f"({row},{col}) overlapping pieces {board_name}")
                return None

        # print(row, col)

        # show_image("piece", piece)

        piece = board[y + crop_amount:y + h - crop_amount, x + crop_amount:x + w - crop_amount]

        piece_color = classify_piece_color(piece)

        config[row][col] = piece_color[0]

    elif row_count == 1:
        for j in range(column_count):
            # cv.rectangle(board, (x, y), (x + w, y + h), (0, 0, 255), 10)
            # cv.line(board, (x + w_piece * j, y), (x + w_piece * j, y + h), (0, 0, 255), 10)

            x_start = x + w_piece * j
            y_start = y
            x_end = x + w_piece * (j + 1)
            y_end = y + h

            x_center = (x_start + x_end) // 2
            y_center = (y_start + y_end) // 2
            col = x_center // 192
            row = y_center // 192

            if config[row][col] != "0":
                if config[row][col] == "1":
                    print("piece placed on 1")
                elif config[row][col] == "2":
                    print("piece placed on 2")
                else:
                    print(f"({row},{col}) overlapping pieces {board_name}")
                    continue

            piece = board[y_start + crop_amount:y_end - crop_amount, x_start + crop_amount:x_end - crop_amount]

            # show_image("piece", piece)

            color = classify_piece_color(piece)

            config[row][col] = color[0]

    elif column_count == 1:
        for i in range(row_count):
            # cv.rectangle(board, (x, y), (x + w, y + h), (0, 0, 255), 10)
            # cv.line(board, (x, y + h_piece * i), (x + w, y + h_piece * i), (0, 0, 255), 10)

            x_start = x
            x_end = x + w
            y_start = y + h_piece * i
            y_end = y + h_piece * (i + 1)

            x_center = (x_start + x_end) // 2
            y_center = (y_start + y_end) // 2

            col = x_center // 192
            row = y_center // 192

            if config[row][col] != "0":
                if config[row][col] == "1":
                    print("piece placed on 1")
                elif config[row][col] == "2":
                    print("piece placed on 2")
                else:
                    print(f"({row},{col}) overlapping pieces {board_name}")
                    continue

            piece = board[y_start + crop_amount:y_end - crop_amount, x_start + crop_amount:x_end - crop_amount]

            # show_image("piece", piece)

            color = classify_piece_color(piece)

            config[row][col] = color[0]


def get_number_pieces(board, config):
    lower1 = np.array([0, 116, 106])
    upper1 = np.array([9, 253, 255])
    lower2 = np.array([160, 116, 106])
    upper2 = np.array([179, 253, 255])

    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)
    mask = cv.bitwise_or(mask1, mask2)

    # show_image("mask", mask)

    contours = get_contours(mask, list_type="list", min_area=1000, max_area=8000, hw_diff_thresh=20)

    for c in contours:
        # x, y, w, h = cv.boundingRect(c)
        # cv.rectangle(board, (x, y), (x+w, y+h), 255, 10)

        row, col, piece = get_cell_index_for_contour(board, c)
        if cv.contourArea(c) >= 5000:
            config[row][col] = "2"
        else:
            config[row][col] = "1"

        # show_image("piece", piece)
    # show_image("b", board)


def resize_image(piece):
    h, w = piece.shape[:2]

    pad_h = 200 - h
    pad_w = 200 - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    resized_img = cv.copyMakeBorder(src=piece, top=top, bottom=bottom, left=left, right=right,
                                    borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))

    return resized_img


def classify_piece_shape(piece):
    # show_image("piece", piece)

    lower, upper = hardcoded_color_ranges["black"]
    lower = np.array(lower)
    upper = np.array(upper)

    piece = cv.cvtColor(piece, cv.COLOR_BGR2HSV)

    mask = cv.inRange(piece, lower, upper)
    mask = ~mask

    mask = resize_image(mask)

    # show_image("mask", mask)

    best_score = -1
    shape = None
    for i, template in enumerate(templates):
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        result = cv.matchTemplate(mask, template, cv.TM_CCOEFF_NORMED)

        _, score, _, _ = cv.minMaxLoc(result)

        if score > best_score:
            shape = shapes[i]
            best_score = score

    return shape


def get_initial_board_config(board):
    # show_image("board", board)

    mask = get_mask(board)
    mask = ~mask
    contours = get_contours(mask, "list", min_area=3000, max_area=25000)

    config = [["0" for _ in range(16)] for _ in range(16)]
    for c in contours:
        row, col, piece = get_cell_index_for_contour(board, c)

        # show_image("piece", piece)

        # resized = resize_image(piece)
        # show_image("resized", resized)

        if config[row][col] == "0":
            color = classify_piece_color(piece)

            # shape = classify_piece_shape(piece)
            # print(shape)

            config[row][col] = color[0]

    # db(config, board_name)

    get_number_pieces(board, config)

    return config


def get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config):
    board_current_copy = board_current.copy()

    mask_diff = get_mask_diff(board_current, board_prev, board_current_name, board_prev_name)
    save_image(mask_diff, "detected_moves/train/diff", board_current_name)

    contours = get_contours(mask_diff, "external", min_area=15000, hw_diff_thresh=100000)
    img = draw_contours(board_current_copy, contours)
    save_image(img, "detected_moves/train/moves", board_current_name)

    for c in contours:
        # if board_current_name == "1_03.jpg":
        #     print(cv.contourArea(c))
        get_pieces_from_contour(board_current, board_current_name, c, config)
