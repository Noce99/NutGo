from Agent import Agent


class HumanAget(Agent):
    def get_move(self, vi, state_value, first_player):
        vi.print_board(state_value, first_player)
        move = vi.human_turn(state_value)
        return move[0], move[1], None
