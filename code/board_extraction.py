import cv2 as cv
import numpy as np
from utilities import show_image, draw_board_corners, save_image, check_display_and_save as cdas


def get_board_corners(edges):
    top_left = top_right = bottom_left = bottom_right = None
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in range(len(contours)):
        if len(contours[i]) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))

                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    return top_left, top_right, bottom_right, bottom_left


def warp_board_perspective(img, width, height, corners):
    top_left, top_right, bottom_right, bottom_left = corners

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)
    return cv.warpPerspective(img, M, (width, height))


def get_canny_edges(img, img_name, displays, saves):
    # TODO: check utilities for inverted images
    r = cv.split(img)[2]
    cdas(r, img_name, displays, saves, "split")

    # TODO: experiment with threshold values
    _, thresh = cv.threshold(r, 80, 150, cv.THRESH_BINARY)
    cdas(thresh, img_name, displays, saves, "thresh")

    # TODO: mess around with iteration counts and order of operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=2)
    cdas(thresh, img_name, displays, saves, "erode")
    thresh = cv.dilate(thresh, kernel, iterations=1)
    cdas(thresh, img_name, displays, saves, "dilate")

    # TODO: experiment with threshold values
    edges = cv.Canny(thresh, 50, 150)
    cdas(edges, img_name, displays, saves, "edges")

    return edges


def extract_board(img, img_name, displays, saves):
    cdas(img, img_name, displays, saves, "original")

    edges = get_canny_edges(img, img_name, displays, saves)

    img_copy = img.copy()
    corners = get_board_corners(edges)
    draw_board_corners(img_copy, corners)
    cdas(img_copy, img_name, displays, saves, "corners")

    # TODO: experiment with values (256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072)
    width = 3072
    height = 3072
    result = warp_board_perspective(img_copy, width, height, corners)

    cdas(result, img_name, displays, saves, "result")

    return result



