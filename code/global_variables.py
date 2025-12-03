import numpy as np
import cv2 as cv

# 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072

# BONUS = True
BONUS = False

DATA_READ_PATH = "../images/train"
# DATA_READ_PATH = "../images/test/fake_test"
# DATA_READ_PATH = "../images/train/bonus"
# DATA_READ_PATH = "../images/test/fake_test/bonus"
# DATA_READ_PATH = "../images/aux-imgs"

BOARDS_SAVE_PATH = "boards/train"
# BOARDS_SAVE_PATH = "boards/test"
# BOARDS_SAVE_PATH = "boards/bonus/train"
# BOARDS_SAVE_PATH = "boards/bonus/test"

MOVES_SAVE_PATH = None
# MOVES_SAVE_PATH = "detected_moves/train"
# MOVES_SAVE_PATH = "detected_moves/test"
# MOVES_SAVE_PATH = "detected_moves/bonus/train"
# MOVES_SAVE_PATH = "detected_moves/bonus/test"

SOLUTION_SAVE_PATH = "solutions"
TEMPLATE_PATH = "templates"
if BONUS:
    SOLUTION_SAVE_PATH += "/bonus"
    TEMPLATE_PATH += "/bonus"

if BONUS:
    CONFIG_ROWS = 215
    CONFIG_COLUMNS = 215

    CONFIG_CENTER_ROW = CONFIG_ROWS // 2
    CONFIG_CENTER_COL = CONFIG_COLUMNS // 2

    EXTEND_CORNERS = 0

    WIDTH_BOARD = 2816
    HEIGHT_BOARD = 2816

    # initialized as None, but defined later in "get_initial_board_config"
    WIDTH_CELL = None
    HEIGHT_CELL = None

    # initialized as None, but defined later in "get_initial_board_config"
    X_ORIGIN = None
    Y_ORIGIN = None

    FREE_CELLS = "0"

else:
    CONFIG_ROWS = 16
    CONFIG_COLUMNS = 16

    CONFIG_CENTER_ROW = None
    CONFIG_CENTER_COL = None

    EXTEND_CORNERS = 50

    WIDTH_BOARD = 2816 + 2 * EXTEND_CORNERS
    HEIGHT_BOARD = 2816 + 2 * EXTEND_CORNERS

    WIDTH_CELL = (WIDTH_BOARD - 2 * EXTEND_CORNERS) // 16
    HEIGHT_CELL = (HEIGHT_BOARD - 2 * EXTEND_CORNERS) // 16

    FREE_CELLS = "0", "1", "2"

BOARD_EXTRACTION_DISPLAYS = {
    # "original": True,
    # "split": True,
    # "thresh": True,
    # "mblur": True,
    # "gblur": True,
    # "erode": True,
    # "dilate": True,
    # "edges": True,
    # "corners": True,
    # "result": True
}
BOARD_EXTRACTION_SAVES = {
    # "original": True,
    # "split": True,
    # "thresh": True,
    # "mblur": True,
    # "gblur": True,
    # "erode": True,
    # "dilate": True,
    # "edges": True,
    # "corners": True,
    "result": True
}

# 0=circle   1=cross   2=diamond   3=square   4=star4   5=star8
# r=red   o=orange   y=yellow   w=white   g=green   b=blue
HARDCODED_COLOR_RANGES = {
    "red": (np.array([170, 100, 100]), np.array([180, 255, 255])),
    "orange": (np.array([6, 46, 97]), np.array([20, 255, 255])),
    "yellow": (np.array([25, 100, 100]), np.array([35, 255, 255])),
    "green": (np.array([40, 87, 0]), np.array([80, 255, 255])),
    "blue": (np.array([100, 100, 100]), np.array([120, 255, 255])),
    "white": (np.array([33, 0, 78]), np.array([156, 88, 255])),
    "black": (np.array([0, 0, 0]), np.array([179, 255, 89]))
}
WHITE_RANGE = (
    np.array([93, 0, 50]),
    np.array([179, 255, 255])
)
COLORS = ("red", "orange", "yellow", "green", "blue", "white")
SHAPES = ("circle", "cross", "diamond", "square", "star4", "star8")

TEMPLATES = []
for i in range(6):
    template = cv.imread(f"{TEMPLATE_PATH}/{i}.jpg")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    TEMPLATES.append(template)
