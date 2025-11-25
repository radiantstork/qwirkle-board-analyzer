import cv2 as cv
import numpy as np
from utilities import show_image, draw_board_corners, save_image, check_display_and_save as cdas
from global_variables import WIDTH_BOARD, HEIGHT_BOARD, EXTEND_CORNERS


def get_board_corners(edges):
    # edges_copy = cv.cvtColor(edges.copy(), cv.COLOR_GRAY2BGR)

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

            # points = np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
            #                                 [possible_bottom_left]])
            area = cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                            [possible_bottom_left]]))
            # if area < 10000:
            #     cv.polylines(edges_copy, points, isClosed=True, color=210, thickness=10)
            if area > max_area:
                max_area = area
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # show_image("test", edges_copy)

    if EXTEND_CORNERS > 0:
        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        centroid = np.mean(corners, axis=0)
        vectors = corners - centroid
        magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1.0
        unit_vectors = vectors / magnitudes
        extended_corners = corners + (EXTEND_CORNERS * unit_vectors)
        rounded_corners = np.round(extended_corners).astype(np.int32)
        new_top_left, new_top_right, new_bottom_right, new_bottom_left = rounded_corners
        top_left = tuple(new_top_left.tolist())
        top_right = tuple(new_top_right.tolist())
        bottom_right = tuple(new_bottom_right.tolist())
        bottom_left = tuple(new_bottom_left.tolist())

    return top_left, top_right, bottom_right, bottom_left


def warp_board_perspective(img, width, height, corners):
    top_left, top_right, bottom_right, bottom_left = corners

    # extend_amount = 50
    # width = width + 2 * extend_amount
    # height = height + 2 * extend_amount

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)
    return cv.warpPerspective(img, M, (width, height))


def get_canny_edges(img, img_name, save_dir, displays, saves):
    img_transformed = cv.split(img)[2]
    cdas(img_transformed, img_name, save_dir, displays, saves, "split")

    # TODO: experiment with threshold values
    _, img_transformed = cv.threshold(img_transformed, 100, 150, cv.THRESH_BINARY_INV)
    cdas(img_transformed, img_name, save_dir, displays, saves, "thresh")

    img_transformed = cv.medianBlur(img_transformed, 7)
    cdas(img_transformed, img_name, save_dir, displays, saves, "mblur")

    # TODO: mess around with iteration counts and order of operations
    kernel = np.ones((5, 5), np.uint8)
    img_transformed = cv.dilate(img_transformed, kernel, iterations=2)
    cdas(img_transformed, img_name, save_dir, displays, saves, "dilate")
    img_transformed = cv.erode(img_transformed, kernel, iterations=1)
    cdas(img_transformed, img_name, save_dir, displays, saves, "erode")

    # invert back
    img_transformed = ~img_transformed

    # TODO: experiment with threshold values
    edges = cv.Canny(img_transformed, 50, 150)
    cdas(edges, img_name, save_dir, displays, saves, "edges")

    # TODO: there is always a horizontal line above the board
    # lsd = cv.createLineSegmentDetector()
    # lines = lsd.detect(edges)[0]
    # lines = lines.reshape(-1, 4)
    # min_y = img.shape[0]
    # for line in lines:
    #     if line[1] < min_y:
    #         min_y = line[1]
    #     if line[3] < min_y:
    #         min_y = line[3]
    # for line in lines:
    #     if min_y - 75 < line[1] < min_y + 75 and min_y - 75 < line[3] < min_y + 75:
    #         cv.line(edges, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 0, 20)
    # show_image("t", edges)
    # line = max(lines, key=lambda x: (np.sqrt((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2)))

    return edges


def extract_board(img, img_name, save_dir, displays, saves):
    cdas(img, img_name, save_dir, displays, saves, "original")

    edges = get_canny_edges(img, img_name, save_dir, displays, saves)

    img_copy = img.copy()
    corners = get_board_corners(edges)
    draw_board_corners(img_copy, corners)
    cdas(img_copy, img_name, save_dir, displays, saves, "corners")

    # TODO: experiment with values (256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072)
    # width = 2816
    # height = 2816

    result = warp_board_perspective(img, WIDTH_BOARD, HEIGHT_BOARD, corners)

    cdas(result, img_name, save_dir, displays, saves, "result")

    return result
