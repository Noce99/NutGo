from monte_carlo_tree_search import MCTS, Node
import random


class NimNode(Node):

    def __init__(self, first_player: bool, state_value):
        """
        :param first_player: True if is a Node for the first player, False if is a Node for the second one
        :param state_value: A unique value that represent the state:
                #
                #
              # #   -> [1, 2, 3, 2, 5]
             ####
            #####
        """
        super().__init__(first_player, state_value)

    def find_children(self):
        for column, elements in enumerate(self.state_value):
            for element_to_remove in range(1, elements+1):
                new_state_value = self.state_value[:]
                new_state_value[column] = elements - element_to_remove
                self.add_children(NimNode(first_player=not self.is_first_player(), state_value=new_state_value))

    def get_random_child(self):
        possible_move = [(column, element_to_remove) for column in range(len(self.state_value))
                         for element_to_remove in range(1, self.state_value[column]+1)]
        random_index = random.randrange(len(possible_move))
        selected_column = possible_move[random_index][0]
        selected_element_to_remove = possible_move[random_index][1]
        new_state_value = self.state_value[:]
        new_state_value[selected_column] = self.state_value[selected_column] - selected_element_to_remove
        return NimNode(first_player=not self.is_first_player(), state_value=new_state_value)

    def get_who_won(self):
        if sum(self.state_value) != 0:
            raise "Asked who who won to a non terminal state!"
        return not self.is_first_player()

    def check_if_terminal(self):
        if sum(self.state_value) == 0:
            return True
        else:
            return False

    def __str__(self):
        str_result = ""
        str_result += "/"
        str_result += "-"*len(self.state_value)
        str_result += "\\\n"
        for raw in range(max(self.state_value)+1, 1, -1):
            str_result += "|"
            for el in self.state_value:
                if raw-el > 1:
                    str_result += " "
                else:
                    str_result += "@"
            str_result += "|\n"
        str_result += "\\"
        str_result += "-" * len(self.state_value)
        str_result += "/"
        return str_result


def get_move_from_human(actual_s):
    while True:
        print("*"*30)
        print(actual_s)
        print(" ", end="")
        for i in range(len(actual_s.state_value)):
            print(i, end="")
        print()
        print("*" * 30)
        selected_column = int(input(f"Select a move! [0; {len(actual_s.state_value)-1}] : "))
        if selected_column in [i for i in range(len(actual_s.state_value))]:
            if actual_s.state_value[selected_column] != 0:
                if actual_s.state_value[selected_column] == 1:
                    element_to_remove = 1
                else:
                    element_to_remove = -1
                while element_to_remove not in [i for i in range(1, actual_s.state_value[selected_column]+1)]:
                    print("*" * 30)
                    print(actual_s)
                    print(" "*(selected_column+1), end="")
                    print("^")
                    print(" " * (selected_column + 1), end="")
                    print("|")
                    print("*" * 30)
                    element_to_remove = int(input(f"Select how many removals! [1; {actual_s.state_value[selected_column]}] : "))
                new_state_value = actual_s.state_value[:]
                new_state_value[selected_column] = actual_s.state_value[selected_column] - element_to_remove
                return NimNode(first_player=not actual_s.is_first_player(), state_value=new_state_value)
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


def play_a_human_vs_computer_game(time, columns):
    human_player_is_first_player = input("Do you want to start? [y or n] : ")
    if human_player_is_first_player.lower() == "y":
        human_player_is_first_player = True
    elif human_player_is_first_player.lower() == "n":
        human_player_is_first_player = False
    else:
        raise "Bad input!"

    actual_state = NimNode(True, [i for i in range(1, columns+1)])
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


play_a_human_vs_computer_game(10, 5)


"""
8 HOURS EXPLORATION:

I have explored for 28800.000002438 seconds!
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
[0] : 0.9980243608015716
/-----\
|    @|
|   @@|
|  @@@|
| @@@@|
| @@@@|
\-----/
[1] : -0.009370581111344154
/-----\
|    @|
|   @@|
|  @@@|
|  @@@|
|@@@@@|
\-----/
[2] : -0.06712062256809338
/-----\
|    @|
|   @@|
|  @@@|
|  @@@|
|@ @@@|
\-----/
[3] : -0.012248417303295242
/-----\
|    @|
|   @@|
|   @@|
| @@@@|
|@@@@@|
\-----/
[4] : -0.04045257284562926
/-----\
|    @|
|   @@|
|   @@|
| @ @@|
|@@@@@|
\-----/
[5] : -0.040207522697795074
/-----\
|    @|
|   @@|
|   @@|
| @ @@|
|@@ @@|
\-----/
[6] : -0.013658645533696076
/-----\
|    @|
|    @|
|  @@@|
| @@@@|
|@@@@@|
\-----/
[7] : -0.022997620935765267
/-----\
|    @|
|    @|
|  @ @|
| @@@@|
|@@@@@|
\-----/
[8] : -0.02506897535914756
/-----\
|    @|
|    @|
|  @ @|
| @@ @|
|@@@@@|
\-----/
[9] : -0.11736334405144695
/-----\
|    @|
|    @|
|  @ @|
| @@ @|
|@@@ @|
\-----/
[10] : -0.009047632235690174
/-----\
|   @@|
|  @@@|
| @@@@|
|@@@@@|
\-----/
[11] : -0.016406483046634186
/-----\
|   @ |
|  @@@|
| @@@@|
|@@@@@|
\-----/
[12] : -0.039951756369666816
/-----\
|   @ |
|  @@ |
| @@@@|
|@@@@@|
\-----/
[13] : -0.08470493777599358
/-----\
|   @ |
|  @@ |
| @@@ |
|@@@@@|
\-----/
[14] : -0.11396226415094339
/-----\
|   @ |
|  @@ |
| @@@ |
|@@@@ |
\-----/
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""