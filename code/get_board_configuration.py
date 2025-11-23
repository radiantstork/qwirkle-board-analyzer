import cv2 as cv
import numpy as np
from utilities import show_image, save_image, draw_board_from_config_matrix as db, draw_contours
from utilities import extend_piece, get_piece_outline
from piece_classification import classify_piece_color, classify_piece_shape
from calculate_score import get_score
from global_variables import hardcoded_color_ranges

mask_cache = {}


def get_mask_of_black_pieces(board):
    lower, upper = hardcoded_color_ranges["black"]

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

    mask = cv.medianBlur(mask, 21)
    # show_image("mblur", mask)

    return mask


def get_contours(mask, list_type, min_area=0, max_area=float("inf"), hw_diff_thresh=15):
    contours, _ = cv.findContours(mask, list_type, cv.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        if min_area <= cv.contourArea(c) <= max_area:
            _, _, w, h = cv.boundingRect(c)
            if abs(w - h) <= hw_diff_thresh:
                filtered_contours.append(c)

    return filtered_contours


def get_position_from_contour(board, x, y, w, h):
    area_of_interest = board[y:y + h, x:x + w]

    x_center = x + w // 2
    y_center = y + h // 2

    col = x_center // 192
    row = y_center // 192

    return row, col, area_of_interest


def get_mask_diff(board_current, board_prev, board_current_name, board_prev_name):
    kernel = np.ones((5, 5), np.uint8)

    board_current = cv.GaussianBlur(board_current, (55, 55), 0)
    board_prev = cv.GaussianBlur(board_prev, (55, 55), 0)
    # show_image("p", board_prev)
    # show_image("c", board_current)

    mask_current = get_mask_of_black_pieces(board_current)
    mask_cache[board_current_name[:-4]] = mask_current
    if board_prev_name[:-4] in mask_cache:
        mask_prev = mask_cache[board_prev_name[:-4]]
    else:
        mask_prev = get_mask_of_black_pieces(board_prev)
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


def update_config(board, board_name, config, x_piece, y_piece, w_piece, h_piece, changes=None):
    row, col, piece = get_position_from_contour(board=board, x=x_piece, y=y_piece, w=w_piece, h=h_piece)

    if config[row][col] not in ("0", "1", "2"):
        print(f"({row},{col}) overlapping pieces {board_name}")
        return None

    # if board_name == "3_09.jpg":
    #     show_image("before extend/outline", piece)

    x_piece, y_piece, w_piece, h_piece = extend_piece(amount=50,
                                                      x_piece=x_piece, y_piece=y_piece,
                                                      w_piece=w_piece, h_piece=h_piece)

    # if board_name == "3_09.jpg":
    #     aux = board[y_piece:y_piece+h_piece, x_piece:x_piece+w_piece]
    #     show_image("extended", aux)

    piece, _, _, _, _ = get_piece_outline(board=board, board_name=board_name,
                                          x_piece=x_piece, y_piece=y_piece, w_piece=w_piece, h_piece=h_piece)

    # if board_name == "1_07.jpg":
    #     show_image("outlined", piece)

    # show_image("piece", piece)

    color = classify_piece_color(piece=piece)
    shape = classify_piece_shape(piece=piece)

    if changes is not None:
        if config[row][col] in ("1", "2"):
            changes.append((row, col, int(config[row][col])))
        else:
            changes.append((row, col, False))

    config[row][col] = f"{shape}{color[0]}"


def get_number_pieces(board, config):
    # TODO: add these hardcoded red values to the dictionary
    lower1 = np.array([0, 116, 106])
    upper1 = np.array([9, 253, 255])
    lower2 = np.array([160, 116, 106])
    upper2 = np.array([179, 253, 255])

    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)
    mask = cv.bitwise_or(mask1, mask2)

    # show_image("mask", mask)

    # TODO: check values
    contours = get_contours(mask=mask, list_type=cv.RETR_LIST, min_area=1000, max_area=8000, hw_diff_thresh=20)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        # cv.rectangle(board, (x, y), (x+w, y+h), 255, 10)

        row, col, piece = get_position_from_contour(board=board, x=x, y=y, w=w, h=h)

        # show_image("piece", piece)

        # TODO: check value
        if cv.contourArea(c) >= 5000:
            config[row][col] = "2"
        else:
            config[row][col] = "1"

    # show_image("b", board)


def get_pieces_from_contour(board, board_name, contour, config, changes, contour_crop):
    x, y, w, h = cv.boundingRect(contour)
    w -= contour_crop
    h -= contour_crop

    row_count = round(h / 192)
    column_count = round(w / 192)

    if row_count == 0 or column_count == 0:
        print(f"0 rows, 0 columns {board_name}")
        return None

    if row_count > 1 and column_count > 1:
        print(f"{row_count} rows {column_count} columns {board_name}")
        return None

    x += contour_crop
    y += contour_crop

    w_piece = w // column_count
    h_piece = h // row_count

    if row_count == 1 and column_count == 1:
        update_config(board=board, board_name=board_name, config=config, changes=changes,
                      x_piece=x, y_piece=y, w_piece=w_piece, h_piece=h_piece)

        # show_image("piece", piece)

    elif row_count == 1:
        for j in range(column_count):
            # cv.rectangle(board, (x, y), (x + w, y + h), (0, 0, 255), 10)
            # cv.line(board, (x + w_piece * j, y), (x + w_piece * j, y + h), (0, 0, 255), 10)

            x_piece = x + w_piece * j

            update_config(board=board, board_name=board_name, config=config, changes=changes,
                          x_piece=x_piece, y_piece=y, w_piece=w_piece, h_piece=h_piece)

            # show_image("piece", piece)

    elif column_count == 1:
        for i in range(row_count):
            # cv.rectangle(board, (x, y), (x + w, y + h), (0, 0, 255), 10)
            # cv.line(board, (x, y + h_piece * i), (x + w, y + h_piece * i), (0, 0, 255), 10)

            y_piece = y + h_piece * i

            update_config(board=board, board_name=board_name, config=config, changes=changes,
                          x_piece=x, y_piece=y_piece, w_piece=w_piece, h_piece=h_piece)

            # show_image("piece", piece)


def get_initial_board_config(board, board_name):
    # show_image("board", board)

    mask = get_mask_of_black_pieces(board=board)
    mask = ~mask

    # save_image(mask, "initial_board_config/train/mask", board_name)

    contours = get_contours(mask, list_type=cv.RETR_LIST, min_area=3000, max_area=25000)

    config = [["0" for _ in range(16)] for _ in range(16)]
    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        update_config(board=board, board_name=board_name, config=config,
                      x_piece=x, y_piece=y, w_piece=w, h_piece=h)

    get_number_pieces(board, config)

    img = db(config)
    save_image(img, "detected_moves/train/config", board_name)

    return config


def get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config):
    board_current_copy = board_current.copy()

    mask_diff = get_mask_diff(board_current=board_current, board_prev=board_prev,
                              board_current_name=board_current_name, board_prev_name=board_prev_name)

    # save_image(mask_diff, "detected_moves/train/diff", board_current_name)

    # TODO: check min_area values
    contours = get_contours(mask=mask_diff, list_type=cv.RETR_EXTERNAL, min_area=15000, hw_diff_thresh=100000)

    img = draw_contours(img=board_current_copy, contours=contours)
    save_image(img, "detected_moves/test/moves", board_current_name)

    changes = []
    for c in contours:
        # TODO: check contour crop values
        get_pieces_from_contour(board=board_current, board_name=board_current_name,
                                contour=c, config=config, changes=changes, contour_crop=30)

    img = db(config)
    # show_image("t", img)
    save_image(img, "detected_moves/test/config", board_current_name)

    score = get_score(config, changes)

    # print(score)
    # show_image(f"{board_current_name}", board_current)

    return score
