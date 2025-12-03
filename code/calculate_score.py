from global_variables import BONUS, CONFIG_ROWS, CONFIG_COLUMNS, FREE_CELLS, CONFIG_CENTER_COL, CONFIG_CENTER_ROW


def up(config, row, col):
    if row < 0 or config[row][col] in FREE_CELLS:
        return 0

    return 1 + up(config, row - 1, col)


def down(config, row, col):
    if row >= CONFIG_ROWS or config[row][col] in FREE_CELLS:
        return 0

    return 1 + down(config, row + 1, col)


def left(config, row, col):
    if col < 0 or config[row][col] in FREE_CELLS:
        return 0

    return 1 + left(config, row, col - 1)


def right(config, row, col):
    if col >= CONFIG_COLUMNS or config[row][col] in FREE_CELLS:
        return 0

    return 1 + right(config, row, col + 1)


def leftmost_position(config, row, col):
    while col >= 0 and config[row][col] not in FREE_CELLS:
        col -= 1

    return col + 1


def highest_position(config, row, col):
    while row >= 0 and config[row][col] not in FREE_CELLS:
        row -= 1

    return row + 1


def check_full_line(score):
    if score == 6:
        return 12

    if score > 1:
        return score

    return 0


def single_piece(config, row, col, bonus_points):
    score = bonus_points

    leftmost_col = leftmost_position(config, row, col)
    aux = right(config, row, leftmost_col)
    score += check_full_line(aux)

    highest_row = highest_position(config, row, col)
    aux = down(config, highest_row, col)
    score += check_full_line(aux)

    return score


def get_coords_and_bonus_points(changes):
    bonus_points = 0
    coords = []
    for row, col, bonus in changes:
        coords.append((row, col))
        bonus_points += bonus

    return coords, bonus_points


def horizontal_pieces(config, changes):
    coords, score = get_coords_and_bonus_points(changes)

    row = coords[0][0]
    leftmost_col = leftmost_position(config, row, coords[0][1])
    col = leftmost_col

    while col < CONFIG_COLUMNS and config[row][col] not in FREE_CELLS:
        if (row, col) in coords:
            highest_row = highest_position(config, row, col)
            aux = down(config, highest_row, col)
            score += check_full_line(aux)

        col += 1

    aux = right(config, row, leftmost_col)
    score += check_full_line(aux)

    return score


def vertical_pieces(config, changes):
    coords, score = get_coords_and_bonus_points(changes)

    col = coords[0][1]
    highest_row = highest_position(config, coords[0][0], col)
    row = highest_row

    while row < CONFIG_ROWS and config[row][col] not in FREE_CELLS:
        if (row, col) in coords:
            leftmost_col = leftmost_position(config, row, col)
            aux = right(config, row, leftmost_col)
            score += check_full_line(aux)

        row += 1

    aux = down(config, highest_row, col)
    score += check_full_line(aux)

    return score


def get_score(config, changes):
    if len(changes) == 0:
        print("COULDN'T CALCULATE SCORE: NO CHANGES ERROR")
        return None

    if BONUS:
        changes = [(-row + CONFIG_CENTER_ROW, col + CONFIG_CENTER_COL, 0) for col, row, _ in changes]

    if len(changes) == 1:
        return single_piece(config, *changes[0])

    if changes[0][0] == changes[1][0]:
        return horizontal_pieces(config, changes)

    return vertical_pieces(config, changes)
