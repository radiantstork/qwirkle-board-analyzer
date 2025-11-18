import cv2 as cv
import numpy as np
from utilities import show_image, save_image, check_display_and_save as cdas


def resize_template(template):
    h, w = template.shape
    delta_h = 200 - h
    delta_w = 200 - w

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    return cv.copyMakeBorder(template, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)


def merge_templates(templates, displays, saves):
    save_dir = "templates/shapes"
    for i in range(6):
        template_name = f"{i}.jpg"

        templates[i] = np.stack(templates[i], axis=0)
        template = np.max(templates[i], axis=0)
        cdas(template, template_name, save_dir, displays, saves, "union")

        kernel = np.ones((5, 5), np.uint8)
        template = cv.erode(template, kernel, iterations=1)
        cdas(template, template_name, save_dir, displays, saves, "erode")


def extract_shapes(img_name, displays, saves):
    board = cv.imread(f"aux_imgs/result/{img_name}")

    hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask = cv.inRange(hsv, lower_red, upper_red)
    # show_image("mask", mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv.contourArea(t) for t in contours])
    max_area = np.max(areas)
    large_area_mask = areas >= (max_area - 15000)
    contours = [contour for contour, condition in zip(contours, large_area_mask) if condition]

    templates = []
    save_dir = "templates/shapes"
    for i, contour in enumerate(contours):
        num = f"0{i}" if i < 10 else f"{i}"
        template_name = f"{num}.jpg"

        x, y, w, h = cv.boundingRect(contour)

        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(board.shape[1], x + w + padding)
        y_end = min(board.shape[0], y + h + padding)

        template = board[y_start:y_end, x_start:x_end]
        cdas(template, template_name, save_dir, displays, saves, "original")

        template = cv.split(template)[2]
        cdas(template, template_name, save_dir, displays, saves, "split")

        _, template = cv.threshold(template, 180, 255, cv.THRESH_BINARY)
        cdas(template, template_name, save_dir, displays, saves, "thresh")

        kernel = np.ones((3, 3), np.uint8)
        template = cv.dilate(template, kernel, iterations=1)
        cdas(template, template_name, save_dir, displays, saves, "dilate")

        template = resize_template(template)
        if i % 3 == 0:
            templates.append([])
        idx = i // 3
        templates[idx].append(template)

    merge_templates(templates, displays, saves)


def extract_numbers():
    # hardcore the data to extract templates from
    img = cv.imread(f"aux_imgs/result/02.jpg")

    # hardcode the color properties to extract pieces of a certain color
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower_red, upper_red)
    # show_image("mask", mask)

    # hardcode area properties to filter only the 1's and 2's
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(contour) >= 1000]

    # hardcore color properties for extraction (again)
    lower_red = np.array([0, 70, 70])
    upper_red = np.array([30, 255, 255])
    i = j = 0
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)

        template = img[y_start:y_end, x_start:x_end]
        show_image("t", template)

        template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
        template = cv.inRange(template_hsv, lower_red, upper_red)
        # show_image("t", template)

        area = cv.contourArea(contour)
        template = resize_template(template)
        # show_image("t", template)
        if area > 5000:
            # save_image(template, "templates/numbers/2", f"{i}.jpg")
            i += 1
        else:
            # save_image(template, "templates/numbers/1", f"{j}.jpg")
            j += 1
