import cv2 as cv
import os
import numpy as np
from global_variables import HARDCODED_COLOR_RANGES, WIDTH_BOARD, HEIGHT_BOARD, BONUS
from global_variables import BOARD_EXTRACTION_DISPLAYS, BOARD_EXTRACTION_SAVES
from global_variables import BOARDS_SAVE_PATH, SOLUTION_SAVE_PATH

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
    if path is None:
        return

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory {path}")

    full_path = f"{path}/{file_name}"
    success = cv.imwrite(full_path, img)

    if not success:
        print(f"Failed to save {full_path}")


def draw_board_corners(img, corners):
    cv.circle(img, corners[0], 20, (0, 0, 255), -1)
    cv.circle(img, corners[1], 20, (0, 0, 255), -1)
    cv.circle(img, corners[2], 20, (0, 0, 255), -1)
    cv.circle(img, corners[3], 20, (0, 0, 255), -1)


def check_display_and_save(img, img_name, key):
    if BOARD_EXTRACTION_DISPLAYS.get(key, False):
        show_image(key, img)

    if BOARD_EXTRACTION_SAVES.get(key, False):
        dir_path = f"{BOARDS_SAVE_PATH}"
        if key != "result":
            dir_path += f"/{key}"

        save_image(img, dir_path, img_name)


def get_contours(mask, list_type, min_area=0, max_area=float("inf"), hw_diff_thresh=float("inf"), min_length=0):
    contours, _ = cv.findContours(mask, list_type, cv.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        if len(c) >= min_length and min_area <= cv.contourArea(c) <= max_area:
            _, _, w, h = cv.boundingRect(c)
            if abs(w - h) <= hw_diff_thresh:
                filtered_contours.append(c)

    return filtered_contours


def draw_contours(img, contours, color=(0, 0, 255)):
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), color, 10)

    return img


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

    offsets = generate_spiral_offsets(radius=50)

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


def save_move(board_name, changes, score, game_index, move_index):
    num = f"0{move_index}" if move_index < 10 else f"{move_index}"
    file_name = f"{game_index}_{num}.txt"

    if BONUS:
        changes.sort(key=lambda x: (x[0], -x[1]))
    else:
        changes.sort(key=lambda x: (int(x[0][:-1]), x[0][-1]))

    with open(f"{SOLUTION_SAVE_PATH}/{file_name}", "w") as g:
        if BONUS:
            for col, row, value in changes:
                g.write(f"{col} {row} {value}\n")

        else:
            for pos, value in changes:
                g.write(f"{pos} {value}\n")

        g.write(f"{score}")

    print(f"({board_name}) saved solution")
