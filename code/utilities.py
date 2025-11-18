import cv2 as cv
import os


def show_image(title, img):
    print(f"Showing image {title}")
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


