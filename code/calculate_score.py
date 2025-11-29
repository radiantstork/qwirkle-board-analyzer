def up(config, row, col):
    if row < 0 or config[row][col] in ("0", "1", "2"):
        return 0

    return 1 + up(config, row - 1, col)


def down(config, row, col):
    if row > 15 or config[row][col] in ("0", "1", "2"):
        return 0

    return 1 + down(config, row + 1, col)


def left(config, row, col):
    if col < 0 or config[row][col] in ("0", "1", "2"):
        return 0

    return 1 + left(config, row, col - 1)


def right(config, row, col):
    if col > 15 or config[row][col] in ("0", "1", "2"):
        return 0

    return 1 + right(config, row, col + 1)


def leftmost_position(config, row, col):
    while col >= 0 and config[row][col] not in ("0", "1", "2"):
        col -= 1

    return col + 1


def highest_position(config, row, col):
    while row >= 0 and config[row][col] not in ("0", "1", "2"):
        row -= 1

    return row + 1


def calculate_bonus(changes):
    changes_coords = []
    total_bonus = 0
    for row, col, bonus in changes:
        changes_coords.append((row, col))
        total_bonus += bonus

    return changes_coords, total_bonus


def check_full_line(score):
    if score == 6:
        return 12

    if score > 1:
        return score

    return 0


def single_piece(config, row, col, bonus):
    score = bonus if bonus else 0

    leftmost_col = leftmost_position(config, row, col)
    aux = right(config, row, leftmost_col)
    score += check_full_line(aux)

    highest_row = highest_position(config, row, col)
    aux = down(config, highest_row, col)
    score += check_full_line(aux)

    return score


def horizontal_pieces(config, changes):
    changes_coords, score = calculate_bonus(changes)

    row = changes[0][0]
    leftmost_col = leftmost_position(config, row, changes[0][1])
    col = leftmost_col

    while col <= 15 and config[row][col] not in ("0", "1", "2"):
        if (row, col) in changes_coords:
            highest_row = highest_position(config, row, col)

            aux = down(config, highest_row, col)
            score += check_full_line(aux)

        col += 1

    aux = right(config, row, leftmost_col)
    score += check_full_line(aux)

    return score


def vertical_pieces(config, changes):
    changes_coords, score = calculate_bonus(changes)

    col = changes_coords[0][1]
    highest_row = highest_position(config, changes_coords[0][0], col)
    row = highest_row

    while row <= 15 and config[row][col] not in ("0", "1", "2"):
        if (row, col) in changes_coords:
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

    if len(changes) == 1:
        return single_piece(config, *changes[0])

    if changes[0][0] == changes[1][0]:
        return horizontal_pieces(config, changes)

    return vertical_pieces(config, changes)


# -----------------------
def right_bonus(config, row, col):
    if col > 29 or config[row][col] == "0":
        return 0

    return 1 + right_bonus(config, row, col + 1)


def down_bonus(config, row, col):
    if row > 29 or config[row][col] == "0":
        return 0

    return 1 + down_bonus(config, row + 1, col)


def leftmost_position_bonus(config, row, col):
    while col >= 0 and config[row][col] != "0":
        col -= 1

    return col + 1


def highest_row_bonus(config, row, col):
    while row >= 0 and config[row][col] != "0":
        row -= 1

    return row + 1


def single_piece_bonus(config, row, col):
    row = 15 + row
    col = 15 + col

    leftmost_col = leftmost_position_bonus(config, row, col)
    aux = right_bonus(config, row, leftmost_col)
    score = check_full_line(aux)

    highest_row = highest_row_bonus(config, row, col)
    aux = down_bonus(config, highest_row, col)
    score += check_full_line(aux)

    return score


def vertical_pieces_bonus(config, changes):
    coords = []
    for change in changes:
        row, col = -change[1], change[0]
        coords.append((row + 15, col + 15))

    col = coords[0][1]
    highest_row = highest_row_bonus(config, coords[0][0], col)
    row = highest_row

    score = 0
    while row <= 29 and config[row][col] != "0":
        if (row, col) in coords:
            leftmost_col = leftmost_position_bonus(config, row, col)

            aux = right_bonus(config, row, leftmost_col)
            score += check_full_line(aux)

        row += 1

    aux = down_bonus(config, highest_row, col)
    score += check_full_line(aux)

    return score


def horizontal_pieces_bonus(config, changes):
    coords = []
    for change in changes:
        row, col = -change[1], change[0]
        coords.append((row + 15, col + 15))

    row = coords[0][0]
    leftmost_col = leftmost_position_bonus(config, row, coords[0][1])
    col = leftmost_col

    score = 0
    while col <= 29 and config[row][col] != "0":
        if (row, col) in coords:
            highest_row = highest_row_bonus(config, row, col)
            aux = down_bonus(config, highest_row, col)
            score += check_full_line(aux)

        col += 1

    aux = right_bonus(config, row, leftmost_col)
    score += check_full_line(aux)

    return score


def get_score_bonus(config, changes):
    # column, -row, value
    if len(changes) == 0:
        print("COULDN'T CALCULATE SCORE: NO CHANGES ERROR")
        return None

    if len(changes) == 1:
        # print("SINGLE PIECE")
        col, row = changes[0][0], -changes[0][1]
        return single_piece_bonus(config, row, col)

    if changes[0][0] == changes[1][0]:
        # print("VERTICAL PIECES")
        return vertical_pieces_bonus(config, changes)

    # print("HORIZONTAL PIECES")
    return horizontal_pieces_bonus(config, changes)
