import copy

import torch

from monte_carlo_tree_search import MCTS, Node
import random
from HexDisplay import Visualizer
from Neural_Networks import HexLinearModel, Trainer


class HexNode(Node):

    def __init__(self, first_player: bool, state_value, added_piece, default_policy=None,
                 probability_of_random_move=1.0):
        """
        :param first_player: True if is a Node for the first player, False if is a Node for the second one
        :param state_value: A unique value that represent the state:
            Supposing the RED (1) player having NE and SW and BLACK (-1) NW and SE
            RED -> FIRST PLAYER
            BLACK -> SECOND PLAYER
              R  R  R  R
            B #  #  1  # B      [[None,          None,           (True, True),   None],
            B -1 #  #  # B -->   [(False, True), None,           None,           None],
            B 1  -1 # -1 B -->   [(True, None),  (False, None),  None,           (False, False)],
            B #  #  1  # B       [None,          None,           (True, False),  None]]
              R  R  R  R
            In the list 'None' means that the cell is empty.
            A tuple means that is full.
            The tuple contain as first element 'True' if it's a first player piece and 'False' if's a second player one.
            The tuple contain as second element 'None' if the piece is NOT connected to BOTH left (top) and right
                (bottom) lines.
            The tuple contain as second element 'True' if the piece is connected to the left (top) line.
            The tuple contain as second element 'False' if the piece is connected to the right (bottom) line.
        :param added_piece: A Tuple containing the raw and the column of the last added piece
        :param default_policy: A function from witch is possible to get the next move during roll-out
        :param probability_of_random_move: Probability of doing a random move during roll-out
        """
        super().__init__(first_player, state_value)
        self.added_piece = added_piece
        self.default_policy = default_policy
        self.probability_of_random_move = probability_of_random_move

        self.size = len(state_value) - 1
        self.final_state = False
        self.winner_first_player = None

    def expand_knowledge(self, r, c, knowledge):
        """
        This function given the coordinate (it doesn't check if they are inside or outside the board) update the
        knowledge (being attached to one side or the other) of the piece supposing (it doesn't check) that there is a
        neighbor of the current player with as knowledge the parameter given to this function.
        This function also change 'self.final_state' if the game is finished!
        :param r: Raw of the Node to be Updated based on the knowledge given
        :param c: Column of the Node to be Updated based on the knowledge given
        :param knowledge: Knowledge given
        :return: None
        """
        if self.state_value[r][c] is not None and self.state_value[r][c][0] is not self.is_first_player():
            # Piece of the same Player that is Playing
            if self.state_value[r][c][1] is None:
                # Piece was not connected now it is
                self.state_value[r][c] = (self.state_value[r][c][0], knowledge)
                self.expand_knowledge_to_neighbor(r, c)
            elif self.state_value[r][c][1] != knowledge:
                # Piece is connected to the other part! The game is finish!
                self.final_state = True
                self.winner_first_player = not self.is_first_player()
                return self.state_value[r][c][1]

    def expand_knowledge_to_neighbor(self, r, c):
        """
        This function select all the possible neighbor taking care of the piece near to the borders and update all the
        existing neighbor. The check about the neighbor player and knowledge is done by 'update_piece'.
        :param r: Raw of Piece from witch the knowledge arrive from.
        :param c: Column of Piece from witch the knowledge arrive from.
        :return: None
        """
        knowledge = self.state_value[r][c][1]
        if r != 0:
            self.expand_knowledge(r - 1, c, knowledge)  # N
            if c != self.size:
                self.expand_knowledge(r - 1, c + 1, knowledge)  # NE
        if r != self.size:
            self.expand_knowledge(r + 1, c, knowledge)  # S
            if c != 0:
                self.expand_knowledge(r + 1, c - 1, knowledge)  # SW
        if c != 0:
            self.expand_knowledge(r, c - 1, knowledge)  # W
        if c != self.size:
            self.expand_knowledge(r, c + 1, knowledge)  # E

    def get_knowledge(self, r, c, actual_knowledge):
        """
        Get the knowledge of the piece at 'r', 'c' and compare it to the one given in 'actual_knowledge' to decide if
        expand it (when actual_knowledge is None) or checking the end of the game.
        :param r: Row of the piece for witch we have to check the knowledge
        :param c: Column of the piece for witch we have to check the knowledge
        :param actual_knowledge: The knowledge to witch we have to compare the one of the piece in 'r', 'c'
        :return: The new knowledge to give to the piece or the unchanged one
        """
        if self.state_value[r][c] is not None and self.state_value[r][c][0] is not self.is_first_player():
            # Piece of the same Player that is Playing
            if self.state_value[r][c][1] is not None:
                # Piece have a knowledge
                if actual_knowledge is None:
                    # The original piece doesn't have a knowledge so give the knowledge to it
                    return self.state_value[r][c][1]
                elif self.state_value[r][c][1] != actual_knowledge:
                    # Piece is connected to the other part! The game is finish!
                    self.final_state = True
                    self.winner_first_player = not self.is_first_player()
                    return self.state_value[r][c][1]
        return None

    def get_knowledge_from_neighbor(self, r, c):
        """
        A function to update the knowledge from the neighbor of a new cell added in 'r', 'c'.
        :param r: Row of the new cell
        :param c: Column of the new cell
        :return: None
        """
        if r != 0:
            new_knowledge = self.get_knowledge(r - 1, c, self.state_value[r][c][1])  # N
            if new_knowledge is not None:
                self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)
            if c != self.size:
                new_knowledge = self.get_knowledge(r - 1, c + 1, self.state_value[r][c][1])  # NE
                if new_knowledge is not None:
                    self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)
        if r != self.size:
            new_knowledge = self.get_knowledge(r + 1, c, self.state_value[r][c][1])  # S
            if new_knowledge is not None:
                self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)
            if c != 0:
                new_knowledge = self.get_knowledge(r + 1, c - 1, self.state_value[r][c][1])  # SW
                if new_knowledge is not None:
                    self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)
        if c != 0:
            new_knowledge = self.get_knowledge(r, c - 1, self.state_value[r][c][1])  # W
            if new_knowledge is not None:
                self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)
        if c != self.size:
            new_knowledge = self.get_knowledge(r, c + 1, self.state_value[r][c][1])  # E
            if new_knowledge is not None:
                self.state_value[r][c] = (self.state_value[r][c][0], new_knowledge)

    def add_piece(self, r, c, piece_is_of_first_player):
        """
        This function add a piece to the board and expand the knowledge of it.
        :param r: Raw to put the Piece
        :param c: Column to put the Piece
        :param piece_is_of_first_player: If the Piece is of the First Player
        :return: None
        """
        if piece_is_of_first_player:
            if r == 0:
                self.state_value[r][c] = (True, True)
                self.expand_knowledge_to_neighbor(r, c)
            elif r == self.size:
                self.state_value[r][c] = (True, False)
                self.expand_knowledge_to_neighbor(r, c)
            else:
                self.state_value[r][c] = (True, None)
                self.get_knowledge_from_neighbor(r, c)
                if self.state_value[r][c][1] is not None:
                    self.expand_knowledge_to_neighbor(r, c)
        else:
            if c == 0:
                self.state_value[r][c] = (False, True)
                self.expand_knowledge_to_neighbor(r, c)
            elif c == self.size:
                self.state_value[r][c] = (False, False)
                self.expand_knowledge_to_neighbor(r, c)
            else:
                self.state_value[r][c] = (False, None)
                self.get_knowledge_from_neighbor(r, c)
                if self.state_value[r][c][1] is not None:
                    self.expand_knowledge_to_neighbor(r, c)

    def find_children(self):
        for r, raw in enumerate(self.state_value):
            for c, cell in enumerate(raw):
                if cell is None:
                    new_node = HexNode(first_player=not self.is_first_player(),
                                       state_value=copy.deepcopy(self.state_value),
                                       added_piece=(r, c),
                                       default_policy=self.default_policy,
                                       probability_of_random_move=self.probability_of_random_move,
                                       )
                    new_node.add_piece(r, c, self.is_first_player())
                    self.add_children(new_node)

    def get_roll_out_child(self):
        if self.default_policy is None or random.random() < self.probability_of_random_move:
            possible_move = [(r, c) for r, raw in enumerate(self.state_value)
                             for c, cell in enumerate(raw) if cell is None]
            random_index = random.randrange(len(possible_move))
            selected_r, selected_c = possible_move[random_index]
        else:
            selected_r, selected_c, _ = self.default_policy(self.state_value, self.first_player)
        new_node = HexNode(first_player=not self.is_first_player(),
                           state_value=copy.deepcopy(self.state_value),
                           added_piece=(selected_r, selected_c),
                           default_policy=self.default_policy,
                           probability_of_random_move=self.probability_of_random_move,
                           )
        new_node.add_piece(selected_r, selected_c, self.is_first_player())
        return new_node

    def get_who_won(self):
        return self.winner_first_player

    def check_if_terminal(self):
        return self.final_state

    def __str__(self):
        return str(self.state_value)


"""
def get_move_from_human(vi, actual_s):
    move = vi.human_turn(actual_s.state_value)
    new_node = HexNode(first_player=not actual_s.is_first_player(),
                       state_value=actual_s.state_value[:],
                       added_piece=(move[0], move[1]),
                       neural_network=actual_s.neural_network,
                       probability_of_random_move=actual_s.probability_of_random_move,
                       )
    new_node.add_piece(move[0], move[1], actual_s.is_first_player())
    return new_node


def get_move_from_nn(nn, actual_s):
    move = nn.predict_move(actual_s.state_value, actual_s.is_first_player, verbose=True)
    new_node = HexNode(first_player=not actual_s.is_first_player(),
                       state_value=actual_s.state_value[:],
                       added_piece=(move[0], move[1]),
                       neural_network=actual_s.neural_network,
                       probability_of_random_move=actual_s.probability_of_random_move,
                       )
    new_node.add_piece(move[0], move[1], actual_s.is_first_player())
    return new_node


def get_move_from_computer(actual_s, time, verbose=False, max_num_of_simulations=-1):
    my_MCS = MCTS(actual_s, time, max_num_of_simulations)
    result = my_MCS.explore(verbose=verbose)
    best_child = result[0][0]
    max_value = result[0][1]
    for i, r in enumerate(result[1:]):
        if not actual_s.is_first_player():
            if r[1] < max_value:
                best_child = r[0]
                max_value = r[1]
        else:
            if r[1] > max_value:
                best_child = r[0]
                max_value = r[1]
    return best_child, result


def play_human_vs_human_game(board_size):
    vi = Visualizer(board_size)
    start_state = HexNode(True, [[None for _ in range(board_size)] for _ in range(board_size)], None)
    state = start_state
    while True:
        state = get_move_from_human(vi, state)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
            else:
                print("Blue Won!")
            vi.finished_game(state.state_value)
            while True:
                pass


def play_human_vs_computer_game(board_size, time):
    vi = Visualizer(board_size)
    start_state = HexNode(True, [[None for _ in range(board_size)] for _ in range(board_size)], None)
    state = start_state
    while True:
        state, _ = get_move_from_computer(state, time)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
            else:
                print("Blue Won!")
            vi.finished_game(state.state_value)
            while True:
                pass
        state = get_move_from_human(vi, state)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
            else:
                print("Blue Won!")
            vi.finished_game(state.state_value)
            while True:
                pass


def play_computer_vs_computer_game(board_size, time, max_num_of_simulations=-1):
    vi = Visualizer(board_size)
    start_state = HexNode(True, [[None for _ in range(board_size)] for _ in range(board_size)], None)
    state = start_state
    while True:
        state, _ = get_move_from_computer(HexNode(True, copy.deepcopy(state.state_value), state.added_piece), time,
                                          max_num_of_simulations=max_num_of_simulations)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
                return 1
            else:
                print("Blue Won!")
                return 2
            vi.finished_game(state.state_value)
            while True:
                pass
        vi.print_board(state.state_value)
        state, _ = get_move_from_computer(HexNode(False, copy.deepcopy(state.state_value), state.added_piece), time,
                                          max_num_of_simulations=max_num_of_simulations)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
                return 1
            else:
                print("Blue Won!")
                return 2
            vi.finished_game(state.state_value)
            while True:
                pass
        vi.print_board(state.state_value)


def test_training_with_data_from_monte_carlo(board_size):
    model = HexLinearModel(board_size)
    coach = Trainer(2, 0.1, 200, model)

    start_state = HexNode(True, [[None for _ in range(board_size)] for _ in range(board_size)], None)
    my_mcs = MCTS(start_state, 5)
    result = my_mcs.explore(verbose=False)
    coach.add_data(start_state.state_value, start_state.first_player, result)
    coach.add_data(start_state.state_value, start_state.first_player, result)
    coach.add_data(start_state.state_value, start_state.first_player, result)
    coach.add_data(start_state.state_value, start_state.first_player, result)
    coach.print_how_much_data()
    coach.train()

    move_from_nn = coach.predict(start_state.state_value, start_state.first_player)
    print(move_from_nn)


def train_neural_network(board_size, prob_of_random_move, each_move_time, num_of_games, max_num_of_simulations=-1):

    return coach


def human_vs_neural_network(nn):
    nn.model.eval()
    board_size = nn.model.board_size
    vi = Visualizer(board_size)
    start_state = HexNode(True, [[None for _ in range(board_size)] for _ in range(board_size)], None)
    state = start_state
    while True:
        state = get_move_from_nn(nn, state)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
            else:
                print("Blue Won!")
            vi.finished_game(state.state_value)
            while True:
                pass
        state = get_move_from_human(vi, state)
        if state.final_state:
            print("Game Finished!")
            if state.winner_first_player:
                print("Red Won!")
            else:
                print("Blue Won!")
            vi.finished_game(state.state_value)
            while True:
                pass


if __name__ == "__main__":
    # test_training_with_data_from_monte_carlo(3)
    # nn = train_neural_network(board_size=4, prob_of_random_move=0.5, each_move_time=1, num_of_games=100,
    #                          max_num_of_simulations=2000)
    # nn.save("Pesi_4x4.w")
    # play_computer_vs_computer_game(4, 1, 2000)
    model = HexLinearModel(4)
    model.load_state_dict(torch.load("Pesi_4x4.w"))
    model.eval()
    coach = Trainer(50, 0.1, 1000, model)
    human_vs_neural_network(coach)
"""

"""
result = play_computer_vs_computer_game(6, 10)
RED: 136, BLUE: 74
result = play_computer_vs_computer_game(5, 10)
RED: 30, BLUE: 6

Code:
    red_won = 0
    blue_won = 0
    while True:
        result = play_computer_vs_computer_game(5, 10)
        if result == 1:
            red_won += 1
        else:
            blue_won += 1
        print(f"RED: {red_won}, BLUE: {blue_won}")
"""
