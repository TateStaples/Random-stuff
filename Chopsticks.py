# october / november 2019

situation_values = {}
hand_name = {0: "left", 1: "right"}


def board_state(board):
    self, other = board
    self_1, self_2 = self
    other_1, other_2 = other
    if self_1 == self_2 == 0:
        return 1
    if other_1 == other_2 == 0:
        return -1
    return 0


def MiniMax(board, ply_depth=0):
    global hand_name, situation_values

    if board in situation_values:
        return situation_values[board]
    if board_state(board) != 0:
        return board_state(board)
    if ply_depth == 20:
        return 0

    if ply_depth % 2 == 0:  # Max
        top = -100
        my_hand = -1
        target_hand = -1
        for count, hand in enumerate(board[0]):
            if hand == 0: continue
            for index, other_hand in enumerate(board[1]):
                new_value = board[1][index] + hand
                if new_value > 4:
                    new_value = 0
                if index == 0:
                    new = (board[0], (new_value, board[1][1]))
                else:
                    new = (board[0], (board[1][0], new_value))
                if new not in situation_values:
                    situation_values[new] = 0
                    not_in = True
                score = MiniMax(new, ply_depth+1)
                if not_in:
                    situation_values[new] = score
                if new not in situation_values:
                    situation_values[new] = score
                if score > top:
                    top = score
                    my_hand = hand_name[count]
                    target_hand = hand_name[index]
                    best_hand = new
        if ply_depth == 0:
            print(f"Use your {my_hand} hand to hit their {target_hand}")
            mine, theirs = best_hand
            print(f"Your hand is {mine} and their hand is {theirs}")

        return top
    else:  # Mini
        mini = 100
        for index, other_hand in enumerate(board[1]):
            if other_hand == 0: continue
            for count, hand in enumerate(board[0]):
                new_value = board[0][count] + other_hand
                if new_value > 4:
                    new_value = 0
                if count == 0:
                    new = ((new_value, board[0][1]), board[1])
                else:
                    new = ((board[0][0], new_value), board[1])
                if new not in situation_values:
                    situation_values[new] = 0
                    not_in = True
                score = MiniMax(new, ply_depth + 1)
                if not_in:
                    situation_values[new] = score
                if score > top:
                    top = score

        return mini


def main():
    while True:
        my_left = int(input("what is your left? "))
        my_right = int(input("what is your right? "))
        other_left = int(input("what is their left? "))
        other_right = int(input("what is their right? "))
        board = ((my_left, my_right), (other_left, other_right))
        MiniMax(board)
        print("tick")


if __name__ == "__main__":
    main()
