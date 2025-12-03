import cv2 as cv
import numpy as np
from utilities import show_image, save_image, draw_contours
from utilities import extend_piece, get_piece_outline, save_move
from utilities import get_contours
from piece_classification import classify_piece_color, classify_piece_shape
from calculate_score import get_score
import global_variables as gv

mask_cache = {}


def get_position_from_contour(x, y, w, h):
    if gv.BONUS:
        x_offset = x - gv.X_ORIGIN
        y_offset = y - gv.Y_ORIGIN

        row = round(y_offset / gv.HEIGHT_CELL)
        col = round(x_offset / gv.WIDTH_CELL)

    else:
        x_center = (x + w // 2) - gv.EXTEND_CORNERS
        y_center = (y + h // 2) - gv.EXTEND_CORNERS

        row = y_center // gv.HEIGHT_CELL
        col = x_center // gv.WIDTH_CELL

    return row, col


def get_mask_diff(current, prev, current_name, prev_name):
    lower, upper = gv.HARDCODED_COLOR_RANGES["black"]
    issues = (
        # current_name,
        # "1_11.jpg",
        # "5_01.jpg"
    )

    mask_prev = None
    if prev_name in mask_cache:
        mask_prev = mask_cache[prev_name]

    if gv.BONUS:
        if mask_prev is None:
            # print(f"({current_name}) CACHE MISS")
            prev = cv.GaussianBlur(prev, (45, 45), 0)
            hsv_prev = cv.cvtColor(prev, cv.COLOR_BGR2HSV)
            mask_prev = cv.inRange(hsv_prev, lower, upper)

        current = cv.GaussianBlur(current, (45, 45), 0)
        hsv_current = cv.cvtColor(current, cv.COLOR_BGR2HSV)
        mask_current = cv.inRange(hsv_current, lower, upper)

        diff = cv.subtract(mask_current, mask_prev)
        diff_mblur = cv.medianBlur(diff, 11)

        diff = diff_mblur

        if current_name in issues:
            show_image(f"({current_name}) diff", diff)
            show_image(f"({current_name}) mblur", diff_mblur)

    else:
        kernel = np.ones((5, 5), np.uint8)

        if mask_prev is None:
            # print(f"({prev_name}) CACHE MISS")
            prev = cv.GaussianBlur(prev, (23, 23), 0)
            hsv_prev = cv.cvtColor(prev, cv.COLOR_BGR2HSV)
            mask_prev = cv.inRange(hsv_prev, lower, upper)

        current = cv.GaussianBlur(current, (23, 23), 0)
        hsv_current = cv.cvtColor(current, cv.COLOR_BGR2HSV)
        mask_current = cv.inRange(hsv_current, lower, upper)

        diff = cv.subtract(mask_current, mask_prev)
        diff_mblur = cv.medianBlur(diff, 17)
        diff_eroded = cv.erode(diff_mblur, kernel, iterations=3)
        diff_dilated = cv.dilate(diff_eroded, kernel, iterations=11)

        diff = diff_dilated

        if current_name in issues:
            show_image(f"({current_name}) diff", diff)
            show_image(f"({current_name}) mblur", diff_mblur)
            show_image(f"({current_name}) erode", diff_eroded)
            show_image(f"({current_name}) dilate", diff_dilated)

    mask_cache[current_name] = mask_current

    if current_name in issues:
        contours = get_contours(diff, cv.RETR_EXTERNAL, min_area=6000)
        bgr = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
        draw_contours(bgr, contours)
        show_image(f"({current_name}) contours", bgr)

    return diff


def update_config(board, board_name, config, x_piece, y_piece, w_piece, h_piece):
    row, col = get_position_from_contour(x=x_piece, y=y_piece, w=w_piece, h=h_piece)

    # POSSIBLE MISDETECTION: there is already a piece in this position
    if (gv.BONUS and config[gv.CONFIG_CENTER_ROW + row][gv.CONFIG_CENTER_COL + col] != "0") or \
            (not gv.BONUS and config[row][col] not in ("0", "1", "2")):
        print(f"({board_name}) PIECE ALREADY EXISTS AT ({row},{col})")
        return None

    # extend what the contour sees
    x_piece, y_piece, w_piece, h_piece = extend_piece(amount=50,
                                                      x_piece=x_piece, y_piece=y_piece,
                                                      w_piece=w_piece, h_piece=h_piece)

    # get the EXACT outline of the piece (for perfect shape classification)
    piece, _, _, _, _ = get_piece_outline(board=board, board_name=board_name,
                                          x_piece=x_piece, y_piece=y_piece, w_piece=w_piece, h_piece=h_piece)

    color = classify_piece_color(piece=piece)
    shape = classify_piece_shape(piece=piece)
    value = f"{shape + 1}{color[0].upper()}"

    if gv.BONUS:
        bonus_score = None
        config[gv.CONFIG_CENTER_ROW + row][gv.CONFIG_CENTER_COL + col] = value

    else:
        bonus_score = int(config[row][col])
        config[row][col] = value

    return row, col, value, bonus_score


def get_pieces_from_contour(board, board_name, contour, config, changes, contour_crop, drawing_board=None):
    x_start, y_start, w, h = cv.boundingRect(contour)
    w -= (2 * contour_crop)
    h -= (2 * contour_crop)

    rows = round(h / gv.HEIGHT_CELL)
    columns = round(w / gv.WIDTH_CELL)

    # error: the contour was too small
    if rows == 0 or columns == 0:
        print(f"({board_name}) 0 ROWS OR 0 COLUMNS, CONTOUR SKIPPED")
        return None

    # error: the contour was too big
    if rows > 1 and columns > 1:
        print(f"({board_name}) {rows} ROWS {columns} COLUMNS, CONTOUR SKIPPED")
        return None

    x_start += contour_crop
    y_start += contour_crop
    w_piece = w // columns
    h_piece = h // rows

    def call_update_config():
        nonlocal changes, x_piece, y_piece, w_piece, h_piece

        aux = update_config(board=board, board_name=board_name, config=config,
                            x_piece=x_piece, y_piece=y_piece, w_piece=w_piece, h_piece=h_piece)

        if aux is None:
            return

        row, column, value, bonus_score = aux
        if gv.BONUS:
            changes.append((column, -row, value))
        else:
            changes.append((row, column, bonus_score))

    # single piece
    if rows == 1 and columns == 1:
        x_piece = x_start
        y_piece = y_start
        call_update_config()

    # horizontal pieces
    elif rows == 1:
        y_piece = y_start
        for j in range(columns):
            x_piece = x_start + w_piece * j
            call_update_config()

    # vertical pieces
    elif columns == 1:
        x_piece = x_start
        for i in range(rows):
            y_piece = y_start + h_piece * i
            call_update_config()

    if drawing_board is not None:
        cv.rectangle(drawing_board, (x_start, y_start), (x_start + w, y_start + h), (0, 0, 255), 10)
    return drawing_board


def get_initial_board_config(board, board_name):
    def get_mask_of_black_pieces():
        nonlocal board

        lower, upper = gv.HARDCODED_COLOR_RANGES["black"]
        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        result = ~cv.inRange(hsv, lower, upper)

        if not gv.BONUS:
            kernel = np.ones((3, 3), np.uint8)
            result = cv.erode(result, kernel, iterations=4)

        return result

    def get_number_pieces():
        nonlocal board, config

        lower1 = np.array([0, 116, 106])
        upper1 = np.array([9, 253, 255])
        lower2 = np.array([160, 116, 106])
        upper2 = np.array([179, 253, 255])

        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)
        mask_numbers = cv.bitwise_or(mask1, mask2)

        number_contours = get_contours(mask=mask_numbers, list_type=cv.RETR_LIST,
                                       min_area=500, max_area=8000, hw_diff_thresh=20)
        draw_contours(highlighted_moves, number_contours)

        for contour in number_contours:
            row, col = get_position_from_contour(*cv.boundingRect(contour))

            if cv.contourArea(contour) >= 4000:
                config[row][col] = "2"
            else:
                config[row][col] = "1"

    mask = get_mask_of_black_pieces()
    if gv.MOVES_SAVE_PATH is not None:
        save_image(mask, f"{gv.MOVES_SAVE_PATH}/mask_diff", board_name)

    highlighted_moves = board.copy()
    config = [["0" for _ in range(gv.CONFIG_COLUMNS)] for _ in range(gv.CONFIG_ROWS)]
    if gv.BONUS:
        # get individual piece contours
        individual_contours = get_contours(mask, cv.RETR_LIST, min_area=5000, max_area=20000, hw_diff_thresh=30)
        num_of_pieces = len(individual_contours)

        # get the overall contour
        mask = ~mask
        full_contour = max(get_contours(mask, cv.RETR_EXTERNAL, min_area=10000),
                           key=cv.contourArea)

        # w_piece = w_contour / num_of_pieces
        # h_piece = h_contour / num_of_pieces
        # final piece dimensions are min(w_piece, h_piece)
        _, _, w, h = cv.boundingRect(full_contour)
        width_aux = w if w < 200 else w // num_of_pieces
        height_aux = h if h < 200 else h // num_of_pieces
        gv.WIDTH_CELL = gv.HEIGHT_CELL = min(width_aux, height_aux)

        top_left_piece_contour = min(individual_contours, key=lambda c: (cv.boundingRect(c)[1] + cv.boundingRect(c)[0]))
        gv.X_ORIGIN, gv.Y_ORIGIN, _, _ = cv.boundingRect(top_left_piece_contour)

        changes = []
        highlighted_moves = get_pieces_from_contour(board=board, board_name=board_name,
                                                    contour=full_contour, contour_crop=0,
                                                    config=config, changes=changes,
                                                    drawing_board=highlighted_moves)

    else:
        # get contours of shaped pieces (input for "update_config" function)
        contours = get_contours(mask, list_type=cv.RETR_LIST, min_area=3000, max_area=25000, hw_diff_thresh=20)
        draw_contours(highlighted_moves, contours)

        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            update_config(board=board, board_name=board_name, config=config,
                          x_piece=x, y_piece=y, w_piece=w, h_piece=h)

        # add the number pieces
        get_number_pieces()
        changes = None

    if gv.MOVES_SAVE_PATH is not None:
        save_image(highlighted_moves, gv.MOVES_SAVE_PATH, board_name)

    return config, changes


def get_intermediary_board_config(current, current_name, prev, prev_name, config):
    # get mask diff between current and previous board
    mask_diff = get_mask_diff(current=current, prev=prev, current_name=current_name, prev_name=prev_name)

    # get contours of newly added pieces
    if gv.BONUS:
        contours = get_contours(mask_diff, cv.RETR_EXTERNAL, min_area=6000)
    else:
        contours = get_contours(mask=mask_diff, list_type=cv.RETR_EXTERNAL, min_area=25000)

    if gv.MOVES_SAVE_PATH is not None:
        mask_bgr = cv.cvtColor(mask_diff, cv.COLOR_GRAY2BGR)
        draw_contours(mask_bgr, contours)
        save_image(mask_bgr, f"{gv.MOVES_SAVE_PATH}/mask_diff", current_name)

    drawing_board = current.copy()
    changes = []

    if gv.BONUS:
        contour_crop = 20
    else:
        contour_crop = 0

    for c in contours:
        get_pieces_from_contour(board=current, board_name=current_name,
                                contour=c, config=config, changes=changes, contour_crop=contour_crop,
                                drawing_board=drawing_board)

    save_image(drawing_board, gv.MOVES_SAVE_PATH, current_name)

    return changes


def get_config(game_index, move_index):
    aux = f"0{move_index}" if move_index < 10 else f"{move_index}"
    prev_name = f"{game_index}_{aux}.jpg"
    prev = cv.imread(f"{gv.BOARDS_SAVE_PATH}/{prev_name}")

    config, changes = get_initial_board_config(prev, prev_name)
    if gv.BONUS:
        score = get_score(config, changes)
        save_move(prev_name, changes, score, game_index, move_index)

    for i in range(move_index + 1, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        current_name = f"{game_index}_{num}.jpg"
        current = cv.imread(f"{gv.BOARDS_SAVE_PATH}/{current_name}")

        changes = get_intermediary_board_config(current, current_name, prev, prev_name, config)

        score = get_score(config, changes)
        if not gv.BONUS:
            changes = [(f"{row + 1}{chr(ord('A') + col)}", config[row][col]) for row, col, _ in changes]

        save_move(current_name, changes, score, game_index, i)

        prev = current
        prev_name = current_name

    return config
