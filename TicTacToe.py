from monte_carlo_tree_search import MCTS, Node
import random


class TicTacToeNode(Node):

    def __init__(self, first_player: bool, state_value):
        """
        :param first_player: True if is a Node for the first player, False if is a Node for the second one
            we arbitrary decide that the first player is "x"
        :param state_value: A unique value that represent the state:
            x|_|o
            _|x|x   --> ["x", "_", "o", "_", "x", "x", "o", "o", "_"]
            o|o|_
        """
        super().__init__(first_player, state_value)
        self.X_won = "?"

    def find_children(self):
        for i, c in enumerate(self.state_value):
            if c == "_":
                new_value = self.state_value[:]
                if self.is_first_player():
                    # x turn
                    new_value[i] = "x"
                else:
                    # o turn
                    new_value[i] = "o"
                self.add_children(TicTacToeNode(first_player=not self.is_first_player(), state_value=new_value))

    def get_random_child(self):
        empty_places = [i for i, c in enumerate(self.state_value) if c == "_"]
        if not empty_places:
            # "empty_places" is empty
            return None
        new_value = self.state_value[:]
        if self.is_first_player():
            # x turn
            new_value[random.choice(empty_places)] = "x"
        else:
            # o turn
            new_value[random.choice(empty_places)] = "o"
        return TicTacToeNode(first_player=not self.is_first_player(), state_value=new_value)

    def get_who_won(self):
        if self.X_won == "?":
            raise "Asking who won before exploring!"
        return self.X_won

    def check_if_terminal(self):
        # check raw
        for raw in range(3):
            first_el = self.state_value[raw * 3]
            if first_el != "_" and self.state_value[raw * 3 + 1] == first_el and self.state_value[raw * 3 + 2] \
                    == first_el:
                # someone won
                if first_el == "x":
                    self.X_won = True
                elif first_el == "o":
                    self.X_won = False
                return True
        # check columns
        for column in range(3):
            first_el = self.state_value[column]
            if first_el != "_" and self.state_value[column + 3] == first_el and self.state_value[column + 6] \
                    == first_el:
                # someone won
                if first_el == "x":
                    self.X_won = True
                elif first_el == "o":
                    self.X_won = False
                return True
        # check diagonal
        first_el = self.state_value[0]
        if first_el != "_" and self.state_value[4] == first_el and self.state_value[8] == first_el:
            # someone won
            if first_el == "x":
                self.X_won = True
            elif first_el == "o":
                self.X_won = False
            return True
        first_el = self.state_value[2]
        if first_el != "_" and self.state_value[4] == first_el and self.state_value[6] == first_el:
            # someone won
            if first_el == "x":
                self.X_won = True
            elif first_el == "o":
                self.X_won = False
            return True
        if "_" not in self.state_value:
            self.X_won = None
            return True
        return False

    def __str__(self):
        str_result = ""
        for raw in range(3):
            for column, content in enumerate(self.state_value[raw * 3:(raw + 1) * 3]):
                if column != 2:
                    str_result += f"{content}|"
                else:
                    str_result += f"{content}\n"
        return str_result


def get_move_from_human(actual_s):
    while True:
        print("*"*30)
        print(actual_s)
        print("*" * 30)
        print(TicTacToeNode(True, ["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
        print("*" * 30)
        move = int(input("Select a move! [0; 8]"))
        if move in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if actual_s.state_value[move] == "_":
                new_value = actual_s.state_value[:]
                if actual_s.is_first_player():
                    new_value[move] = "x"
                else:
                    new_value[move] = "o"
                return TicTacToeNode(first_player=not actual_s.is_first_player(), state_value=new_value)
        print("Wrong move stupid human!")


def get_move_from_computer(actual_s, time, verbose=False):
    my_MCS = MCTS(actual_s, time)
    result = my_MCS.explore(verbose=verbose)
    best_child = result[0][0]
    max_value = result[0][1]
    print("@"*30)
    print(f"[{0}] : {result[0][1]}")
    print(result[0][0])
    for i, r in enumerate(result[1:]):
        if not actual_s.is_first_player():
            if r[1] < max_value:
                best_child = r[0]
                max_value = r[1]
        else:
            if r[1] > max_value:
                best_child = r[0]
                max_value = r[1]

        print(f"[{i+1}] : {r[1]}")
        print(r[0])
    print("@" * 30)
    return best_child


def play_a_human_vs_computer_game(time):
    human_player_is_first_player = input("Witch player do you want to be? [x or o]")
    if human_player_is_first_player.lower() == "x":
        human_player_is_first_player = True
    elif human_player_is_first_player.lower() == "o":
        human_player_is_first_player = False
    else:
        raise "Bad input!"

    actual_state = TicTacToeNode(True, ["_", "_", "_", "_", "_", "_", "_", "_", "_"])
    while True:
        if human_player_is_first_player:
            actual_state = get_move_from_human(actual_state)
            if actual_state.is_terminal():
                the_winner = actual_state.get_who_won()
                break
            actual_state = get_move_from_computer(actual_state, time)
            if actual_state.is_terminal():
                the_winner = actual_state.get_who_won()
                break
        else:
            actual_state = get_move_from_computer(actual_state, time)
            if actual_state.is_terminal():
                the_winner = actual_state.get_who_won()
                break
            actual_state = get_move_from_human(actual_state)
            if actual_state.is_terminal():
                the_winner = actual_state.get_who_won()
                break

    if the_winner is None:
        print("No one won!")
    elif the_winner == human_player_is_first_player:
        print("The human won!")
    else:
        print("The computer won! Stupid Human!")


# A = TicTacToeNode(True, ["x", "o", "x", "_", "x", "_", "o", "_", "o"])
# A = TicTacToeNode(False, ["o", "x", "o", "_", "o", "_", "x", "_", "x"])
# get_move_from_computer(A, 1, False)

play_a_human_vs_computer_game(1)
