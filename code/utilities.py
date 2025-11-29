import cv2 as cv
import os
import numpy as np
from global_variables import HARDCODED_COLOR_RANGES, WIDTH_BOARD, HEIGHT_BOARD, WIDTH_CELL, HEIGHT_CELL, BONUS

COLOR_MAP = {
    "r": "red",
    "o": "orange",
    "y": "yellow",
    "g": "green",
    "b": "blue",
    "w": "white"
}
SHAPE_MAP = {
    "0": "circle",
    "1": "cross",
    "2": "diamond",
    "3": "square",
    "4": "star4",
    "5": "star8"
}


def show_image(title, img):
    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_image(img, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory {path}")

    full_path = f"{path}/{file_name}"
    success = cv.imwrite(full_path, img)

    if not success:
        print(f"Failed to save {full_path}")


def verify_game_and_move_index(game_index, move_index, data_type):
    if BONUS:
        if game_index != 1:
            print(f"unknown game index (bonus-{data_type}-{game_index})")
            return False

        if move_index < 1 or move_index > 20:
            print(f"unknown move index (bonus-{data_type}-{move_index})")
            return False

    else:
        if data_type == "test":
            if game_index != 1:
                print(f"unknown game index test-{game_index}")
                return False

        elif data_type == "train":
            if game_index < 1 or game_index > 5:
                print(f"unknown game index train-{game_index}")
                return False

        else:
            print(f"unknown data type ({data_type})")
            return False

        if move_index < 0 or move_index > 20:
            print(f"unknown move index {data_type}-{move_index}")
            return False

    return True


def draw_board_corners(img, corners):
    cv.circle(img, corners[0], 20, (0, 0, 255), -1)
    cv.circle(img, corners[1], 20, (0, 0, 255), -1)
    cv.circle(img, corners[2], 20, (0, 0, 255), -1)
    cv.circle(img, corners[3], 20, (0, 0, 255), -1)


def check_display_and_save(img, img_name, save_dir, displays, saves, key):
    if displays.get(key):
        show_image(key, img)

    if saves.get(key):
        dir_path = f"{save_dir}"
        if key != "result":
            dir_path += f"/{key}"

        save_image(img, dir_path, img_name)


def draw_board_from_config_matrix(config):
    GRID_SIZE = 16
    CELL_SIZE = 102
    BORDER_WIDTH = 2
    CONTENT_SIZE = 100
    RADIUS = CONTENT_SIZE // 2
    LINE_THICKNESS = 10

    GRAY_BG = (186, 186, 186)
    BLACK_BG = (0, 0, 0)
    COLOR_MAP = {
        "r": (0, 0, 255),
        "o": (0, 165, 255),
        "y": (0, 255, 255),
        "g": (0, 255, 0),
        "b": (255, 0, 0),
        "w": (255, 255, 255),
        "1": (0, 0, 255),
        "2": (0, 0, 255)
    }

    img = np.zeros((1632, 1632, 3), dtype=np.uint8)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = config[row][col]

            x_start = col * CELL_SIZE
            y_start = row * CELL_SIZE

            x_content_start = x_start + BORDER_WIDTH
            y_content_start = y_start + BORDER_WIDTH
            x_content_end = x_start + CELL_SIZE - BORDER_WIDTH
            y_content_end = y_start + CELL_SIZE - BORDER_WIDTH

            background_color = BLACK_BG if cell != "0" else GRAY_BG

            cv.rectangle(img,
                         (x_content_start, y_content_start), (x_content_end, y_content_end),
                         background_color, -1)

            if cell != "0":
                x_center = x_content_start + RADIUS
                y_center = y_content_start + RADIUS

                if cell == "1" or cell == "2":
                    color = COLOR_MAP[cell]
                else:
                    color = COLOR_MAP[cell[1]]

                if cell == "1":
                    font_scale = RADIUS / 20.0
                    thickness = max(1, int(RADIUS / 8))
                    font = cv.FONT_HERSHEY_SIMPLEX

                    text_size = cv.getTextSize("1", font, font_scale, thickness)[0]
                    text_x = x_center - text_size[0] // 2
                    text_y = y_center + text_size[1] // 2

                    cv.putText(img, "1",
                               (text_x, text_y),
                               font, font_scale, color, thickness, cv.LINE_AA)

                elif cell == "2":
                    font_scale = RADIUS / 20.0
                    thickness = max(1, int(RADIUS / 8))
                    font = cv.FONT_HERSHEY_SIMPLEX

                    text_size = cv.getTextSize("2", font, font_scale, thickness)[0]
                    text_x = x_center - text_size[0] // 2
                    text_y = y_center + text_size[1] // 2

                    cv.putText(img, "2",
                               (text_x, text_y),
                               font, font_scale, color, thickness, cv.LINE_AA)

                elif cell[0] == "0":
                    cv.circle(img, (x_center, y_center), RADIUS, color, -1)

                elif cell[0] == "1":
                    cv.line(img,
                            (x_center - RADIUS, y_center), (x_center + RADIUS, y_center),
                            color, LINE_THICKNESS)
                    cv.line(img,
                            (x_center, y_center - RADIUS), (x_center, y_center + RADIUS),
                            color, LINE_THICKNESS)

                elif cell[0] == "2":
                    pts = np.array([
                        (x_center, y_center - RADIUS),
                        (x_center + RADIUS, y_center),
                        (x_center, y_center + RADIUS),
                        (x_center - RADIUS, y_center)
                    ], np.int32)
                    cv.fillPoly(img, [pts], color)

                elif cell[0] == "3":
                    cv.rectangle(img,
                                 (x_center - RADIUS, y_center - RADIUS),
                                 (x_center + RADIUS, y_center + RADIUS),
                                 color, -1)

                elif cell[0] == "4" or cell[0] == "5":
                    diag_radius = int(RADIUS / 1.414)

                    cv.line(img,
                            (x_center - diag_radius, y_center - diag_radius),
                            (x_center + diag_radius, y_center + diag_radius),
                            color, LINE_THICKNESS)

                    cv.line(img,
                            (x_center + diag_radius, y_center - diag_radius),
                            (x_center - diag_radius, y_center + diag_radius),
                            color, LINE_THICKNESS)

                if cell[0] == "5":
                    cv.line(img,
                            (x_center - RADIUS, y_center),
                            (x_center + RADIUS, y_center),
                            color, LINE_THICKNESS)

                    cv.line(img,
                            (x_center, y_center - RADIUS),
                            (x_center, y_center + RADIUS),
                            color, LINE_THICKNESS)

    return img


def get_contours(mask, list_type, min_area=0, max_area=float("inf"), hw_diff_thresh=float("inf")):
    contours, _ = cv.findContours(mask, list_type, cv.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        if min_area <= cv.contourArea(c) <= max_area:
            _, _, w, h = cv.boundingRect(c)
            if abs(w - h) <= hw_diff_thresh:
                filtered_contours.append(c)

    return filtered_contours


def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), 255, 10)

    return img


def generate_spiral_offsets(radius):
    offsets = [(0, 0)]
    for k in range(1, radius + 1):
        # top edge (left to right)
        for x in range(-k, k):
            offsets.append((-k, x))

        # right edge (top to bottom)
        for y in range(-k, k):
            offsets.append((y, k))

        # bottom edge (right to left)
        for x in range(k, -k, -1):
            offsets.append((k, x))

        # left edge (bottom to top)
        for y in range(k, -k, -1):
            offsets.append((y, -k))

    return offsets


def get_piece_info(config, row, col):
    piece = config[row][col]

    if len(piece) == 1:
        if piece == "0":
            info = "empty cell"
        else:
            info = f"number {piece}"
    else:
        shape = SHAPE_MAP[piece[0]]
        color = COLOR_MAP[piece[1]]
        info = f"{color} {shape}"

    return info


def compare_configs(config, label):
    mistake_count = 0
    mistakes = []
    for i in range(16):
        for j in range(16):
            if config[i][j] != label[i][j]:
                mistakes.append((i, j))

                info = get_piece_info(config, i, j)
                label_info = get_piece_info(label, i, j)

                print(f"({i},{j}): {info} should be {label_info}")

                mistake_count += 1

    print(f"TOTAL: {mistake_count} MISTAKES")

    return mistakes


def resize_image(img, new_w, new_h):
    h, w = img.shape[:2]

    if h > new_h:
        h_crop = h - new_h
        h_start = h_crop // 2
        h_end = h - (h_crop - h_start)
        img = img[h_start:h_end, :]

    elif h < new_h:
        pad_h = new_h - h
        top = pad_h // 2
        bottom = pad_h - top
        img = cv.copyMakeBorder(src=img, top=top, bottom=bottom, left=0, right=0,
                                borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))

    if w > new_w:
        w_crop = w - new_w
        w_start = w_crop // 2
        w_end = w - (w_crop - w_start)
        img = img[:, w_start:w_end]

    elif w < new_w:
        pad_w = new_w - w
        left = pad_w // 2
        right = pad_w - left
        img = cv.copyMakeBorder(src=img, top=0, bottom=0, left=left, right=right,
                                borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))

    return img


def extend_piece(amount, x_piece, y_piece, w_piece, h_piece):
    max_extend_left = min(amount, x_piece)
    max_extend_top = min(amount, y_piece)
    max_extend_right = min(amount, WIDTH_BOARD - x_piece - w_piece)
    max_extend_bottom = min(amount, HEIGHT_BOARD - y_piece - h_piece)

    amount = min(max_extend_top, max_extend_right, max_extend_bottom, max_extend_left)

    x_piece -= amount
    y_piece -= amount
    w_piece += (2 * amount)
    h_piece += (2 * amount)

    return x_piece, y_piece, w_piece, h_piece


def get_piece_outline(board, board_name, x_piece, y_piece, w_piece, h_piece, erode=True):
    lower, upper = HARDCODED_COLOR_RANGES["black"]

    piece = board[y_piece:y_piece + h_piece, x_piece:x_piece + w_piece]

    # show_image("p", piece)

    hsv = cv.cvtColor(piece, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = ~mask

    # show_image("outline mask", mask)

    if erode:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)

    # show_image("eroded mask", mask)

    # TODO: check radius values, window_size=2*radius+1
    offsets = generate_spiral_offsets(radius=10)

    x_center = w_piece // 2
    y_center = h_piece // 2
    seed_point = None
    for y, x in offsets:
        x_candidate = x_center + x
        y_candidate = y_center + y

        if 0 <= x_candidate < w_piece and 0 <= y_candidate < h_piece:
            if mask[y_candidate, x_candidate] == 255:
                seed_point = (x_candidate, y_candidate)
                break

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    if seed_point is None:
        print(f"couldn't find seed point {board_name}")
        show_image(f"{board_name}", mask)
        return None, None, None, None, None

    flood_mask = np.zeros((h_piece + 2, w_piece + 2), dtype=np.uint8)
    fill_color = (0, 255, 0)
    cv.floodFill(mask, flood_mask, (seed_point[0], seed_point[1]), fill_color, flags=4)

    # show_image("flooded mask", mask)

    mask = cv.inRange(mask, fill_color, fill_color)

    points = cv.findNonZero(mask)

    x_new, y_new, w_new, h_new = cv.boundingRect(points)

    return piece[y_new:y_new + h_new, x_new:x_new + w_new], (x_piece + x_new), (y_piece + y_new), w_new, h_new


def get_piece_from_position(board, board_name, row, col, erode=True):
    y = row * HEIGHT_CELL
    x = col * WIDTH_CELL
    w = WIDTH_CELL
    h = HEIGHT_CELL

    # piece = board[y:y+h, x:x+w]
    # show_image("p", piece)

    x, y, w, h = extend_piece(amount=50,
                              x_piece=x, y_piece=y,
                              w_piece=w, h_piece=h)

    piece, _, _, _, _ = get_piece_outline(board=board, board_name=board_name,
                                          x_piece=x, y_piece=y, w_piece=w, h_piece=h,
                                          erode=erode)

    return piece


def verify_move(changes, score, game_index, move_index, data_type):
    if not verify_game_and_move_index(game_index, move_index, data_type):
        return False

    sol_path = f"../images/{data_type}"
    if data_type == "test":
        sol_path += "/fake_test/ground-truth"
    if BONUS:
        sol_path += "/bonus"

    num = f"0{move_index}" if move_index < 10 else f"{move_index}"
    file_name = f"{game_index}_{num}.txt"
    f = open(f"{sol_path}/{file_name}")

    lines = f.readlines()
    f.close()

    changes_label = []
    for line in lines[:-1]:
        line = line.rstrip("\n").split()
        changes_label.append((int(line[0]), int(line[1]), line[2]))
    score_label = int(lines[-1])

    changes_label.sort(key=lambda x: (x[0], x[1]))
    changes.sort(key=lambda x: (x[0], x[1]))
    if changes != changes_label:
        print(f"({game_index}_{num}.jpg) WRONG MOVE DETECTED")
        print(changes)
        print(changes_label)
        return False

    if score != score_label:
        print(f"({game_index}_{num}.jpg) WRONG SCORE")
        print(score, score_label)
        return False

    print(f"({game_index}_{num}.jpg) correct")
    return True
