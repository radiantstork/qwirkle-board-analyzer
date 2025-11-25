import cv2 as cv
import numpy as np
from utilities import show_image, save_image, draw_board_from_config_matrix as db, draw_contours
from utilities import extend_piece, get_piece_outline
from piece_classification import classify_piece_color, classify_piece_shape
from calculate_score import get_score
from global_variables import hardcoded_color_ranges, WIDTH_CELL, HEIGHT_CELL, EXTEND_CORNERS

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

    x_center = (x + w // 2) - EXTEND_CORNERS
    y_center = (y + h // 2) - EXTEND_CORNERS

    col = x_center // WIDTH_CELL
    row = y_center // HEIGHT_CELL

    return row, col, area_of_interest


def get_mask_diff(board_current, board_prev, board_current_name, board_prev_name):
    # 3072x3072: 55x55 gblur, diff, mblur 23, erode 5x5 3, mblur 11, dilate 5x5 18
    lower, upper = hardcoded_color_ranges["black"]
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((11, 11), np.uint8)

    board_current = cv.GaussianBlur(board_current, (85, 85), 0)
    board_prev = cv.GaussianBlur(board_prev, (85, 85), 0)
    # show_image("p", board_prev)
    # show_image("c", board_current)

    hsv_prev = cv.cvtColor(board_prev, cv.COLOR_BGR2HSV)
    mask_prev = cv.inRange(hsv_prev, lower, upper)

    hsv_current = cv.cvtColor(board_current, cv.COLOR_BGR2HSV)
    mask_current = cv.inRange(hsv_current, lower, upper)

    # show_image("mask_prev", mask_prev)
    # show_image("mask_current", mask_current)

    mask_diff = np.subtract(mask_current, mask_prev)

    # show_image("mask_diff", mask_diff)

    mask_diff = cv.medianBlur(mask_diff, 11)

    # show_image("diff", mask_diff)

    mask_diff = cv.erode(mask_diff, kernel_erode, iterations=3)

    # show_image("erode", mask_diff)

    mask_diff = cv.dilate(mask_diff, kernel_dilate, iterations=9)

    # show_image("dilate", mask_diff)

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
    # 3072x3072 values
    # contours = get_contours(mask=mask, list_type=cv.RETR_LIST, min_area=1000, max_area=8000, hw_diff_thresh=20)

    # 2816x2816 values
    contours = get_contours(mask=mask, list_type=cv.RETR_LIST, min_area=500, max_area=8000, hw_diff_thresh=20)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        # cv.rectangle(board, (x, y), (x+w, y+h), 255, 10)

        row, col, piece = get_position_from_contour(board=board, x=x, y=y, w=w, h=h)

        # show_image("piece", piece)

        # print(cv.contourArea(c))
        # show_image("t", piece)

        # TODO: check value
        # 3072x3072 value: 5000
        if cv.contourArea(c) >= 4000:
            config[row][col] = "2"
        else:
            config[row][col] = "1"

    # show_image("b", board)


def get_pieces_from_contour(board, board_name, contour, config, changes, contour_crop, drawing_board=None):
    x, y, w, h = cv.boundingRect(contour)
    w -= (2 * contour_crop)
    h -= (2 * contour_crop)

    row_count = round(h / HEIGHT_CELL)
    column_count = round(w / WIDTH_CELL)

    if row_count == 0 or column_count == 0:
        print(f"0 rows, 0 columns {board_name}")
        return None

    if row_count > 1 and column_count > 1:
        print(f"{row_count} rows {column_count} columns {board_name}")
        return None

    x += contour_crop
    y += contour_crop

    if drawing_board is not None:
        cv.rectangle(drawing_board, (x, y), (x+w, y+h), (0, 0, 255), 10)

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

    # img = db(config)
    # save_image(img, "detected_moves/train/config", board_name)

    return config


def get_intermediary_board_config(board_current, board_current_name, board_prev, board_prev_name, config,
                                  data_type="train"):
    save_path = f"detected_moves/{data_type}"

    board_current_copy = board_current.copy()

    mask_diff = get_mask_diff(board_current=board_current, board_prev=board_prev,
                              board_current_name=board_current_name, board_prev_name=board_prev_name)

    # TODO: check min_area values
    contours = get_contours(mask=mask_diff, list_type=cv.RETR_EXTERNAL, min_area=30000, hw_diff_thresh=100000)

    mask_bgr = cv.cvtColor(mask_diff, cv.COLOR_GRAY2BGR)
    draw_contours(mask_bgr, contours)
    save_image(mask_bgr, f"{save_path}/mask_diff", board_current_name)

    changes = []
    for c in contours:
        # TODO: check contour crop values
        get_pieces_from_contour(board=board_current, board_name=board_current_name,
                                contour=c, config=config, changes=changes, contour_crop=30,
                                drawing_board=board_current_copy)

    save_image(board_current_copy, f"{save_path}", board_current_name)

    # show_image("t", img)
    # save_image(img, f"detected_moves/{data_type}/config", board_current_name)

    score = get_score(config, changes)

    # print(score)
    # show_image(f"{board_current_name}", board_current)

    return score
