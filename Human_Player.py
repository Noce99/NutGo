from Agent import Agent
from Agent_DRL import get_child_selected_by_nn, find_obvious_moves, from_state_value_to_tensor


class HumanAget(Agent):
    def get_move(self, vi, state, first_player):
        # JUST FOR TEST ##########################################################################
        tensor_state_value = from_state_value_to_tensor(state, first_player)
        obv_r, obv_c = find_obvious_moves(tensor_state_value, first_player)
        if obv_r is not None:
            print(f"An obv move is: [{obv_r}, {obv_c}]")
        # JUST FOR TEST ##########################################################################
        vi.print_board(state.state_value, first_player)
        move = vi.human_turn(state)
        best_child = get_child_selected_by_nn(move[0], move[1], state)
        return move[0], move[1], None, best_child
