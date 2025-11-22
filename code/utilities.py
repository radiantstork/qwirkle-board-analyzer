import cv2 as cv
import os
import numpy as np


def show_image(title, img):
    # print(f"Showing image {title}")
    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_board_corners(img, corners):
    cv.circle(img, corners[0], 20, (0, 0, 255), -1)
    cv.circle(img, corners[1], 20, (0, 0, 255), -1)
    cv.circle(img, corners[2], 20, (0, 0, 255), -1)
    cv.circle(img, corners[3], 20, (0, 0, 255), -1)


def save_image(img, directory_path, file_name):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory {directory_path}")

    full_path = f"{directory_path}/{file_name}"
    success = cv.imwrite(full_path, img)

    if not success:
        print(f"Failed to save {full_path}")
    # else:
    #     print(f"Saved file: {full_path}")


def check_display_and_save(img, img_name, save_dir, displays, saves, key):
    if displays.get(key):
        show_image(key, img)

    if saves.get(key):
        dir_path = f"{save_dir}/{key}"
        save_image(img, dir_path, img_name)


def load_templates():
    shapes = []
    for i in range(6):
        img = cv.imread(f"templates/shapes/{i}.jpg")
        shapes.append(img)

    return shapes


def draw_board_from_config_matrix(config):
    GRID_SIZE = 16
    CELL_SIZE = 102
    BORDER_WIDTH = 2
    CONTENT_SIZE = 100
    RADIUS = CONTENT_SIZE // 2

    WHITE_BG = (255, 255, 255)
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

            background_color = BLACK_BG if cell != "0" else WHITE_BG
            cv.rectangle(
                img,
                (x_content_start, y_content_start),
                (x_content_end, y_content_end),
                background_color,
                -1
            )

            if cell != "0":
                x_center = x_content_start + RADIUS
                y_center = y_content_start + RADIUS

                color = COLOR_MAP[cell]

                if cell == "1":
                    font_scale = RADIUS / 20.0
                    thickness = max(1, int(RADIUS / 8))
                    font = cv.FONT_HERSHEY_SIMPLEX

                    text_size = cv.getTextSize("1", font, font_scale, thickness)[0]
                    text_x = x_center - text_size[0] // 2
                    text_y = y_center + text_size[1] // 2

                    cv.putText(img, "1", (text_x, text_y), font, font_scale, color, thickness, cv.LINE_AA)

                elif cell == "2":
                    font_scale = RADIUS / 20.0
                    thickness = max(1, int(RADIUS / 8))
                    font = cv.FONT_HERSHEY_SIMPLEX

                    text_size = cv.getTextSize("2", font, font_scale, thickness)[0]
                    text_x = x_center - text_size[0] // 2
                    text_y = y_center + text_size[1] // 2

                    cv.putText(img, "2", (text_x, text_y), font, font_scale, color, thickness, cv.LINE_AA)

                else:
                    cv.circle(img, (x_center, y_center), RADIUS, color, -1)

    return img


def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), 255, 10)

    return img
