import cv2 as cv
import numpy as np
from utilities import draw_board_corners, check_display_and_save as cdas
from utilities import get_contours, show_image
from global_variables import BONUS, WIDTH_BOARD, HEIGHT_BOARD, EXTEND_CORNERS, WHITE_RANGE
from global_variables import BOARD_EXTRACTION_DISPLAYS, BOARD_EXTRACTION_SAVES
from global_variables import DATA_READ_PATH


def get_candidate_points(contour):
    top_left = bottom_right = None
    for point in contour.squeeze():
        if top_left is None or point[0] + point[1] < top_left[0] + top_left[1]:
            top_left = point

        if bottom_right is None or point[0] + point[1] > bottom_right[0] + bottom_right[1]:
            bottom_right = point

    diff = np.diff(contour.squeeze(), axis=1)
    top_right = contour.squeeze()[np.argmin(diff)]
    bottom_left = contour.squeeze()[np.argmax(diff)]

    return top_left, top_right, bottom_right, bottom_left


def extend_board_corners(top_left, top_right, bottom_right, bottom_left):
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    centroid = np.mean(corners, axis=0)

    vectors = corners - centroid

    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1.0

    unit_vectors = vectors / magnitudes

    corners = corners + (EXTEND_CORNERS * unit_vectors)
    corners = np.round(corners).astype(np.int32)

    new_top_left, new_top_right, new_bottom_right, new_bottom_left = corners
    top_left = tuple(new_top_left.tolist())
    top_right = tuple(new_top_right.tolist())
    bottom_right = tuple(new_bottom_right.tolist())
    bottom_left = tuple(new_bottom_left.tolist())

    return top_left, top_right, bottom_right, bottom_left


def get_board_corners(edges):
    top_left = top_right = bottom_left = bottom_right = None
    contours = get_contours(edges, cv.RETR_EXTERNAL, hw_diff_thresh=500, min_length=3)

    max_area = 0
    for c in contours:
        top_left_aux, top_right_aux, bottom_right_aux, bottom_left_aux = get_candidate_points(c)
        points = np.array([[top_left_aux], [top_right_aux], [bottom_right_aux], [bottom_left_aux]])
        area = cv.contourArea(points)

        if area > max_area:
            max_area = area
            top_left, top_right = top_left_aux, top_right_aux
            bottom_right, bottom_left = bottom_right_aux, bottom_left_aux

    if EXTEND_CORNERS > 0:
        return extend_board_corners(top_left, top_right, bottom_right, bottom_left)

    return top_left, top_right, bottom_right, bottom_left


def get_board_corners_bonus(mask):
    h, w = mask.shape[:2]

    top_left = (0, np.max(np.where(mask[:, 0] == 255)[0]))
    top_right = (w - 1, np.max(np.where(mask[:, w - 1] == 255)[0]))
    bottom_right = (w - 1, h - 1)
    bottom_left = (0, h - 1)

    return top_left, top_right, bottom_right, bottom_left


def warp_board_perspective(img, corners):
    puzzle = np.array(corners, dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [WIDTH_BOARD, 0], [WIDTH_BOARD, HEIGHT_BOARD], [0, HEIGHT_BOARD]],
                                     dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)
    return cv.warpPerspective(img, M, (WIDTH_BOARD, HEIGHT_BOARD))


def get_canny_edges(img, img_name):
    img = cv.split(img)[2]
    cdas(img, img_name, "split")

    _, img = cv.threshold(img, 100, 150, cv.THRESH_BINARY_INV)
    cdas(img, img_name, "thresh")

    img = cv.medianBlur(img, 7)
    cdas(img, img_name, "mblur")

    kernel = np.ones((3, 3), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    cdas(img, img_name, "dilate")

    edges = cv.Canny(img, 50, 150)
    cdas(edges, img_name, "edges")

    return edges


def extract_board(img, img_name):
    cdas(img, img_name, "original")

    if BONUS:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, WHITE_RANGE[0], WHITE_RANGE[1])
        corners = get_board_corners_bonus(mask)
        board = warp_board_perspective(img, corners)

    else:
        edges = get_canny_edges(img, img_name)

        corners = get_board_corners(edges)
        if BOARD_EXTRACTION_DISPLAYS.get("corners", False) or BOARD_EXTRACTION_SAVES.get("corners", False):
            img_copy = img.copy()
            draw_board_corners(img_copy, corners)
            cdas(img_copy, img_name, "corners")

        board = warp_board_perspective(img, corners)

    cdas(board, img_name, "result")
    print(f"({img_name}) extraction finished")
    return board


def get_boards(game_index, move_index):
    for i in range(move_index, 21):
        num = f"0{i}" if i < 10 else f"{i}"

        img_name = f"{game_index}_{num}.jpg"
        img = cv.imread(f"{DATA_READ_PATH}/{img_name}")

        extract_board(img, img_name)


