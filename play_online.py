# Import and initialize your own actor
import time

from Agent_DRL import HexAgent, get_child_selected_by_nn
from Hex import HexNode
from ActorClient import ActorClient
from HexDisplay import Visualizer


# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):
    def __init__(self, *args, **kwargs):
        super(MyClient, self).__init__(*args, **kwargs)
        self.my_state = None
        self.board_size = None
        self.player_number = None
        self.I_need_to_start = None
        self.actor = None
        self.last_server_state = None

        self.visualizer = False
        if not self.visualizer:
            self.vi = None
        else:
            self.vi = Visualizer(7)

    def find_last_move(self, state):
        # print(self.last_server_state)
        # print(state)
        for i in range(1, len(self.last_server_state)):
            if self.last_server_state[i] != state[i]:
                return (i - 1) // 7, (i - 1) - ((i - 1) // 7) * 7

    def handle_get_action(self, state):
        if self.I_need_to_start:
            self.I_need_to_start = False
            r, c, probability, best_child = self.actor.get_move(self.vi, self.my_state, self.my_state.is_first_player())
            self.my_state = best_child
            self.last_server_state[r * 7 + c + 1] = self.player_number
            # print(f"My Move: ({r}, {c})")
            # print(self.last_server_state)
            return r, c

        last_row, last_col = self.find_last_move(state)
        # print(f"Opponent Move: ({last_row}, {last_col})")
        self.last_server_state = state

        self.my_state = get_child_selected_by_nn(last_row, last_col, self.my_state)

        if self.visualizer:
            self.vi.print_board(self.my_state.state_value, self.my_state.first_player)

        r, c, probability, best_child = self.actor.get_move(self.vi, self.my_state, self.my_state.is_first_player())
        self.my_state = best_child

        if self.visualizer:
            self.vi.print_board(self.my_state.state_value, self.my_state.first_player)

        # print(self.last_server_state)
        # print(f"My Move: ({r}, {c})")
        self.last_server_state[r * 7 + c + 1] = self.player_number
        # print(self.last_server_state)
        return r, c

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        self.board_size = game_params[0]
        for player in player_map:
            if player[0] == unique_id:
                self.player_number = player[1]
        print("HANDLE SERIES START")
        print(f"unique_id=\"{unique_id}\", series_id=\"{series_id}\", player_map=\"{player_map}\","
              f" num_games=\"{num_games}\", game_params=\"{game_params}\"")

    def handle_game_start(self, start_player):
        self.my_state = HexNode(first_player=(start_player == 1),
                                state_value=[[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                                default_policy=None,
                                probability_of_random_move=0)
        self.actor = HexAgent(board_size=self.board_size, batch_size=64, learning_rate=1, max_num_of_data=4000)
        self.actor.load_weight("7x7_Nut_v_2_0")
        if start_player == self.player_number:
            self.I_need_to_start = True
        else:
            self.I_need_to_start = False
        self.last_server_state = [1] + [0 for _ in range(49)]
        print("HANDLE GAME START")
        print(f"start_player=\"{start_player}\"")

    def handle_game_over(self, winner, end_state):
        print("HANDLE GAME OVER")
        print(f"winner=\"{winner}\"")

    def handle_series_over(self, stats):
        print("HANDLE SERIES OVER")
        print(f"stats=\"{stats}\"")

    def handle_tournament_over(self, score):
        print("HANDLE TOURNAMENT OVER")
        print(f"score=\"{score}\"")


def show_a_state(state):
    vi = Visualizer(7)
    state_value = [[None for _ in range(7)] for _ in range(7)]
    for i in range(1, len(state)):
        row = (i - 1) // 7
        col = (i - 1) - ((i - 1) // 7) * 7
        if state[i] == 1:
            state_value[row][col] = (True, None)
        elif state[i] == 2:
            state_value[row][col] = (False, None)
    vi.finished_game(state_value)


if __name__ == '__main__':
    #show_a_state([1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0])
    client = MyClient(auth="1ae1aff183ac40cba865df8d199437b0", qualify=False)
    client.run()  # mode="league")

"""
HANDLE GAME START
start_player="2"
Move: (0, 2)
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (4, 4)
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (1, 5)
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (0, 3)
[1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (0, 1)
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (4, 0)
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (5, 2)
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (0, 6)
[1, 0, 2, 0, 2, 0, 0, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 0, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (0, 5)
[1, 0, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (5, 0)
[1, 0, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (2, 3)
[1, 0, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 0, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (5, 1)
[1, 0, 2, 0, 2, 0, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (3, 2)
[1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 2, 0, 0, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (4, 2)
[1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Move: (0, 2)

"""
