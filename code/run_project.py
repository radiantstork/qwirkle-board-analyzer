from get_board_configuration import get_config
from board_extraction import get_boards
from global_variables import SOLUTION_SAVE_PATH
import os

os.makedirs(SOLUTION_SAVE_PATH, exist_ok=True)
os.makedirs(f"{SOLUTION_SAVE_PATH}/bonus", exist_ok=True)

for i in range(1, 2):
    get_boards(i, 0)
    get_config(i, 0)
