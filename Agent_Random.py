import random

from Agent import Agent


class RandomAgent(Agent):
    def get_move(self, vi, state_value, first_player=None):
        possible_move = [(r, c) for r, raw in enumerate(state_value)
                         for c, cell in enumerate(raw) if cell is None]
        random_index = random.randrange(len(possible_move))
        selected_r, selected_c = possible_move[random_index]
        return selected_r, selected_c, None
