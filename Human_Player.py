from Agent import Agent
from Agent_DRL import get_child_selected_by_nn

class HumanAget(Agent):
    def get_move(self, vi, state, first_player):
        vi.print_board(state.state_value, first_player)
        move = vi.human_turn(state)
        best_child = get_child_selected_by_nn(move[0], move[1], state)
        return move[0], move[1], None, best_child
