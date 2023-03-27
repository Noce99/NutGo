# Import and initialize your own actor
from Agent_DRL import HexAgent
from Hex import HexNode
from ActorClient import ActorClient
from HexDisplay import Visualizer


# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):
    def __init__(self, *args, **kwargs):
        super(MyClient, self).__init__(*args, **kwargs)
        self.my_state = HexNode(True, [[None for _ in range(7)] for _ in range(7)], None)
        self.actor = HexAgent(board_size=7, batch_size=64, learning_rate=1, max_num_of_data=4000)
        self.last_server_state = [1] + [0 for _ in range(49)]
        self.vi = Visualizer(7)

    def find_last_move(self, state):
        # print(self.last_server_state)
        # print(state)
        for i in range(1, len(self.last_server_state)):
            if self.last_server_state[i] != state[i]:
                return (i-1) // 7, (i-1) - ((i-1) // 7) * 7

    def handle_get_action(self, state):
        last_row, last_col = self.find_last_move(state)
        # print(f"Move: ({last_row}, {last_col})")
        self.last_server_state = state

        self.my_state = HexNode(first_player=False,
                                state_value=self.my_state.state_value[:],
                                added_piece=(last_row, last_col))
        self.my_state.add_piece(last_row, last_col, True)

        self.vi.print_board(self.my_state.state_value, self.my_state.first_player)

        row, col, _ = self.actor.get_move(None, self.my_state.state_value, self.my_state.is_first_player())
        row = int(row)
        col = int(col)
        self.my_state = HexNode(first_player=True,
                                state_value=self.my_state.state_value[:],
                                added_piece=(row, col))
        self.my_state.add_piece(row, col, False)

        self.vi.print_board(self.my_state.state_value, self.my_state.first_player)

        self.last_server_state[row*7 + col + 1] = 1
        if self.my_state.final_state:
            print(self.last_server_state)
            self.vi.finished_game(self.my_state.state_value)
        return row, col


if __name__ == '__main__':
    client = MyClient(auth="1ae1aff183ac40cba865df8d199437b0", qualify=False)
    client.run()