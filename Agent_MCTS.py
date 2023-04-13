import torch

from Agent import Agent
from Agent_DRL import from_result_to_tensor
from Hex import HexNode
from monte_carlo_tree_search import MCTS


class MCTSAget(Agent):

    def __init__(self, time, max_num_of_simulations):
        self.time = time
        self.max_num_of_simulations = max_num_of_simulations

    def get_move(self, vi, state, first_player):
        start_node = HexNode(first_player, state.state_value, None)
        my_MCS = MCTS(start_node, self.time, self.max_num_of_simulations)
        result = my_MCS.explore(verbose=False)
        # for r, p in result:
        #    print(r.added_piece, p)
        probability = from_result_to_tensor(len(state.state_value), result)
        # print(torch.reshape(probability, (4, 4)))
        best_child = result[0][0]
        max_value = result[0][1]
        for i, r in enumerate(result[1:]):
            if r[1] > max_value:
                best_child = r[0]
                max_value = r[1]
        return best_child.added_piece[0], best_child.added_piece[1], probability, best_child
