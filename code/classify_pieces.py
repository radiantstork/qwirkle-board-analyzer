import cv2 as cv
import numpy as np
from utilities import show_image, save_image


def extract_pieces(board, board_name):
    for i in range(16):
        for j in range(16):
            y_top = i * 192
            y_bottom = (i + 1) * 192
            x_left = j * 192
            x_right = (j + 1) * 192

            piece = board[y_top: y_bottom, x_left: x_right]
            piece_idx = i * 16 + j
            save_image(piece, f"extracted_pieces/{board_name[:-4]}", f"{piece_idx}.jpg")