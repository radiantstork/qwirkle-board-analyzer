import cv2 as cv
import numpy as np
from utilities import show_image, save_image, draw_board_from_config_matrix as db, draw_contours
from utilities import extend_piece, get_piece_outline, verify_game_and_move_index, verify_move
from utilities import compare_configs, get_piece_info, get_piece_from_position, get_contours
from piece_classification import classify_piece_color, classify_piece_shape
from calculate_score import get_score, get_score_bonus
import global_variables

mask_cache = {}


def get_position_from_contour(x, y, w, h):
    if global_variables.BONUS:
        x_offset = x - global_variables.X_ORIGIN
        y_offset = y - global_variables.Y_ORIGIN

        row = round(y_offset / global_variables.HEIGHT_CELL)
        column = round(x_offset / global_variables.WIDTH_CELL)

        return row, column

    else:
        x_center = (x + w // 2) - global_variables.EXTEND_CORNERS
        y_center = (y + h // 2) - global_variables.EXTEND_CORNERS

        col = x_center // global_variables.WIDTH_CELL
        row = y_center // global_variables.HEIGHT_CELL

        return row, col


def get_mask_diff(current, prev, current_name, prev_name):
    lower, upper = global_variables.HARDCODED_COLOR_RANGES["black"]
    issues = (
        # current_name,

        # "1_11.jpg",

        # "5_01.jpg"
    )

    def normal_transformations():
        nonlocal current, prev
        kernel = np.ones((5, 5), np.uint8)

        if prev_name not in mask_cache:
            prev = cv.GaussianBlur(prev, (23, 23), 0)
            hsv_prev = cv.cvtColor(prev, cv.COLOR_BGR2HSV)
            mask_prev = cv.inRange(hsv_prev, lower, upper)
            # print(f"({prev_name}) cache miss")
        else:
            mask_prev = mask_cache[prev_name]
            # print(f"({prev_name}) cache hit")

        current = cv.GaussianBlur(current, (23, 23), 0)
        hsv_current = cv.cvtColor(current, cv.COLOR_BGR2HSV)
        mask_current = cv.inRange(hsv_current, lower, upper)
        mask_cache[current_name] = mask_current

        mask_diff = cv.subtract(mask_current, mask_prev)
        diff_mblur = cv.medianBlur(mask_diff, 17)
        diff_eroded = cv.erode(diff_mblur, kernel, iterations=3)
        diff_dilated = cv.dilate(diff_eroded, kernel, iterations=11)

        mask_diff = diff_dilated

        if current_name in issues:
            show_image(f"({current_name}) diff", mask_diff)
            show_image(f"({current_name}) mblur", diff_mblur)
            show_image(f"({current_name}) erode", diff_eroded)
            show_image(f"({current_name}) dilate", diff_dilated)

            contours = get_contours(mask_diff, cv.RETR_EXTERNAL, min_area=6000)
            bgr = cv.cvtColor(mask_diff, cv.COLOR_GRAY2BGR)
            draw_contours(bgr, contours)
            show_image(f"({current_name}) contours", bgr)

        return mask_diff

    def bonus_transformations():
        nonlocal current, prev

        current = cv.GaussianBlur(current, (45, 45), 0)
        prev = cv.GaussianBlur(prev, (45, 45), 0)

        hsv_prev = cv.cvtColor(prev, cv.COLOR_BGR2HSV)
        mask_prev = cv.inRange(hsv_prev, lower, upper)

        hsv_current = cv.cvtColor(current, cv.COLOR_BGR2HSV)
        mask_current = cv.inRange(hsv_current, lower, upper)

        if current_name in issues:
            show_image("prev", mask_prev)
            show_image("current", mask_current)

        diff = cv.subtract(mask_current, mask_prev)
        if current_name in issues:
            show_image(f"({current_name}) diff", diff)

        diff = cv.medianBlur(diff, 11)
        if current_name in issues:
            show_image(f"({current_name}) mblur", diff)

        # kernel = np.ones((5, 5), np.uint8)
        # diff = cv.dilate(diff, kernel, iterations=6)
        # if current_name in issues:
        #     show_image(f"({current_name}) dilate", diff)

        if current_name in issues:
            contours, _ = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv.contourArea(c) >= 6000]
            bgr = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
            draw_contours(bgr, contours)
            show_image(f"({current_name}) contours", bgr)

        return diff

    if global_variables.BONUS:
        return bonus_transformations()

    return normal_transformations()


def update_config(board, board_name, config, x_piece, y_piece, w_piece, h_piece):
    row, col = get_position_from_contour(x=x_piece, y=y_piece, w=w_piece, h=h_piece)

    # POSSIBLE MISDETECTION: there is already a piece in this position
    if (global_variables.BONUS and config[15 + row][15 + col] != "0") or \
            (not global_variables.BONUS and config[row][col] not in ("0", "1", "2")):
        print(f"({row},{col}) overlapping pieces {board_name}")
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

    if global_variables.BONUS:
        config[15 + row][15 + col] = value
    else:
        config[row][col] = value

    return row, col, value


def get_pieces_from_contour(board, board_name, contour, config, changes, contour_crop, drawing_board=None):
    x_start, y_start, w, h = cv.boundingRect(contour)
    w -= (2 * contour_crop)
    h -= (2 * contour_crop)

    rows = round(h / global_variables.HEIGHT_CELL)
    columns = round(w / global_variables.WIDTH_CELL)

    # error: the contour was too small
    if rows == 0 or columns == 0:
        print(f"({board_name}) 0 rows OR 0 columns error")
        return None

    # error: the contour was too big
    if rows > 1 and columns > 1:
        print(f"({board_name}) {rows} rows {columns} columns error")
        return None

    x_start += contour_crop
    y_start += contour_crop
    w_piece = w // columns
    h_piece = h // rows

    # single piece
    if rows == 1 and columns == 1:
        row, column, value = update_config(board=board, board_name=board_name, config=config,
                                           x_piece=x_start, y_piece=y_start, w_piece=w_piece, h_piece=h_piece)

        if global_variables.BONUS:
            changes.append((column, -row, value))

    # horizontal pieces
    elif rows == 1:
        for j in range(columns):
            x_piece = x_start + w_piece * j

            row, column, value = update_config(board=board, board_name=board_name, config=config,
                                               x_piece=x_piece, y_piece=y_start, w_piece=w_piece, h_piece=h_piece)

            if global_variables.BONUS:
                changes.append((column, -row, value))

    # vertical pieces
    elif columns == 1:
        for i in range(rows):
            y_piece = y_start + h_piece * i

            row, column, value = update_config(board=board, board_name=board_name, config=config,
                                               x_piece=x_start, y_piece=y_piece, w_piece=w_piece, h_piece=h_piece)

            if global_variables.BONUS:
                changes.append((column, -row, value))

    if drawing_board is not None:
        cv.rectangle(drawing_board, (x_start, y_start), (x_start + w, y_start + h), (0, 0, 255), 10)
    return drawing_board


def get_initial_board_config(board, board_name, data_type):
    def get_mask_of_black_pieces():
        nonlocal board

        lower, upper = global_variables.HARDCODED_COLOR_RANGES["black"]

        if global_variables.BONUS:
            hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower, upper)

        else:
            kernel = np.ones((3, 3), np.uint8)
            hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            mask = ~ cv.inRange(hsv, lower, upper)
            mask = cv.erode(mask, kernel, iterations=4)

        return mask

    def get_number_pieces():
        nonlocal board, config
        # TODO: add these hardcoded red values to the dictionary
        lower1 = np.array([0, 116, 106])
        upper1 = np.array([9, 253, 255])
        lower2 = np.array([160, 116, 106])
        upper2 = np.array([179, 253, 255])

        hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)
        mask = cv.bitwise_or(mask1, mask2)

        # TODO: check values
        # 3072x3072 values
        # contours = get_contours(mask=mask, list_type=cv.RETR_LIST, min_area=1000, max_area=8000, hw_diff_thresh=20)

        # 2816x2816 values
        contours = get_contours(mask=mask, list_type=cv.RETR_LIST, min_area=500, max_area=8000, hw_diff_thresh=20)

        for c in contours:
            x, y, w, h = cv.boundingRect(c)

            row, col = get_position_from_contour(x=x, y=y, w=w, h=h)

            # TODO: check value
            # 3072x3072 value: 5000
            if cv.contourArea(c) >= 4000:
                config[row][col] = "2"
            else:
                config[row][col] = "1"

    save_path = "detected_moves"
    if global_variables.BONUS:
        save_path += "/bonus"
    save_path += f"/{data_type}"

    mask = get_mask_of_black_pieces()
    save_image(mask, f"{save_path}/mask_diff", board_name)
    if global_variables.BONUS:
        config = [["0" for _ in range(30)] for _ in range(30)]

        # get individual piece contours
        mask = ~mask
        individual_contours = get_contours(mask, cv.RETR_LIST, min_area=5000, max_area=20000, hw_diff_thresh=30)
        num_of_pieces = len(individual_contours)

        # get the overall contour
        mask = ~mask
        full_contour = max(get_contours(mask, cv.RETR_EXTERNAL, min_area=10000, hw_diff_thresh=100000),
                           key=cv.contourArea)

        # w_piece = w_contour / num_of_pieces
        # h_piece = h_contour / num_of_pieces
        # final piece dimensions are min(w_piece, h_piece)
        _, _, w, h = cv.boundingRect(full_contour)
        width_aux = w if w < 200 else w // num_of_pieces
        height_aux = h if h < 200 else h // num_of_pieces
        global_variables.WIDTH_CELL = global_variables.HEIGHT_CELL = min(width_aux, height_aux)

        top_left_piece_contour = min(individual_contours, key=lambda c: (cv.boundingRect(c)[1], cv.boundingRect(c)[0]))
        global_variables.X_ORIGIN, global_variables.Y_ORIGIN, _, _ = cv.boundingRect(top_left_piece_contour)

        changes = []
        highlighted_moves = board.copy()
        highlighted_moves = get_pieces_from_contour(board=board, board_name=board_name,
                                                    contour=full_contour, contour_crop=0,
                                                    config=config, changes=changes,
                                                    drawing_board=highlighted_moves)

    else:
        # get contours of shaped pieces (input for "update_config" function)
        contours = get_contours(mask, list_type=cv.RETR_LIST, min_area=3000, max_area=25000, hw_diff_thresh=20)
        config = [["0" for _ in range(16)] for _ in range(16)]
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            update_config(board=board, board_name=board_name, config=config,
                          x_piece=x, y_piece=y, w_piece=w, h_piece=h)

        # add the number pieces
        get_number_pieces()
        changes = None
        # save_image(detected_moves, save_path, board_name)

    return config, changes


def get_intermediary_board_config(current, current_name, prev, prev_name, config, data_type):
    save_path = "detected_moves"
    if global_variables.BONUS:
        save_path += "/bonus"
    save_path += f"/{data_type}"

    # get mask diff between current and previous board
    mask_diff = get_mask_diff(current=current, prev=prev, current_name=current_name, prev_name=prev_name)

    # TODO: check min_area values
    # get contours of newly added pieces
    if global_variables.BONUS:
        contours = get_contours(mask_diff, cv.RETR_EXTERNAL, min_area=6000)
    else:
        contours = get_contours(mask=mask_diff, list_type=cv.RETR_EXTERNAL, min_area=25000)

    mask_bgr = cv.cvtColor(mask_diff, cv.COLOR_GRAY2BGR)
    draw_contours(mask_bgr, contours)
    save_image(mask_bgr, f"{save_path}/mask_diff", current_name)

    drawing_board = current.copy()
    changes = []
    if global_variables.BONUS:
        contour_crop = 20
    else:
        contour_crop = 0
    for c in contours:
        # TODO: check contour crop values
        get_pieces_from_contour(board=current, board_name=current_name,
                                contour=c, config=config, changes=changes, contour_crop=contour_crop,
                                drawing_board=drawing_board)

    save_image(drawing_board, save_path, current_name)

    return changes

    # TODO score


def get_config(game_index, move_index, data_type):
    if not verify_game_and_move_index(game_index, move_index, data_type):
        return None

    read_path = "boards"
    save_path = "detected_moves"
    if global_variables.BONUS:
        read_path += "/bonus"
        save_path += "/bonus"
    read_path += f"/{data_type}"
    save_path += f"/{data_type}"

    aux = f"0{move_index}" if move_index < 10 else f"{move_index}"
    prev_name = f"{game_index}_{aux}.jpg"
    prev = cv.imread(f"{read_path}/{prev_name}")

    config, changes = get_initial_board_config(prev, prev_name, data_type)
    score = get_score_bonus(config, changes)
    verify_move(changes, score, game_index, move_index, data_type)
    for i in range(move_index + 1, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        current_name = f"{game_index}_{num}.jpg"
        current = cv.imread(f"{read_path}/{current_name}")

        changes = get_intermediary_board_config(current, current_name, prev, prev_name, config, data_type)
        score = get_score_bonus(config, changes)
        verify_move(changes, score, game_index, i, data_type)

        # score = get_intermediary_board_config(current, current_name, prev, prev_name, config, data_type)
        # scores.append(score)

        prev = current
        prev_name = current_name

    return config

    if move_index == 0:
        print(f"GAME {game_index}")

        key = "test" if data_type == "test" else f"train_{game_index}"
        config_label = global_variables.CONFIG_LABELS_MAP[key]
        scores_label = global_variables.SCORE_LABELS_MAP[key]

        mistakes = compare_configs(config, config_label)
        if tuple(scores) == scores_label:
            print("scores are correct")
        else:
            print("scores are wrong ")

        for row, col in mistakes:
            info = get_piece_info(config, row, col)
            piece = get_piece_from_position(prev, prev_name, row, col)
            print(f"({row},{col}) {info}")
            show_image("real piece", piece)

