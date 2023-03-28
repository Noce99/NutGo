import time

import torch

from HexDisplay import Visualizer
from Agent_DRL import HexAgent
from Hex import HexNode
from Human_Player import HumanAget
from Agent_Random import RandomAgent
from monte_carlo_tree_search import MCTS
from Agent_MCTS import MCTSAget


class HexGame:
    def __init__(self, board_size):
        self.board_size = board_size

    def play_a_match(self, agent1, agent2, wait=0):
        vi = Visualizer(self.board_size)
        start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)], None)
        state = start_state
        while True:
            r, c, probability = agent1.get_move(vi, state.state_value, state.is_first_player())
            state = HexNode(first_player=not state.is_first_player(),
                            state_value=state.state_value[:],
                            added_piece=(r, c))
            state.add_piece(r, c, not state.is_first_player())  # This not is ok!
            time.sleep(wait)
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
            vi.print_board(state.state_value, state.first_player)
            if probability is not None:
                vi.print_probability(probability)
            r, c, probability = agent2.get_move(vi, state.state_value, state.is_first_player())
            state = HexNode(first_player=not state.is_first_player(),
                            state_value=state.state_value[:],
                            added_piece=(r, c))
            state.add_piece(r, c, not state.is_first_player())  # This not is ok!
            time.sleep(wait)
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
            vi.print_board(state.state_value, state.first_player)
            if probability is not None:
                vi.print_probability(probability, offset=30)


if __name__ == "__main__":
    dlr_agent = HexAgent(board_size=7, batch_size=32, learning_rate=3, max_num_of_data=5000)
    # dlr_agent.load_weight("7x7_something.w")
    #dlr_agent.save_dataset()
    dlr_agent.load_dataset("7x7_2023_03_28_21_56_13")
    dlr_agent.single_training(300)
    dlr_agent.trainer.plot_loss()
    #dlr_agent.evaluate_with_random_model(matches=100)
    #dlr_agent.train_while_playing(epochs=100, time_limit=5, simulations_limit=2000)
    # dlr_agent.evaluate_model()
    #dlr_agent.manually_evaluation()
    #dlr_agent.save_weight()

    #a_game = HexGame(7)
    #human = HumanAget()
    #a_game.play_a_match(human, MCTSAget(time=10, max_num_of_simulations=2000), wait=0)
    vi = Visualizer(7)
    #vi.show_dataset(dlr_agent.dataset)
    vi.visual_evaluation(dlr_agent.dataset, dlr_agent.trainer)

"""
The sum of probability on free cell was 0 but I'm smart so I select a random move:
Asked to predict a move but no free cell available!
tensor_state_value
tensor([[ 1., -1.,  0.,  1.,  1., -1., -1.,  0., -1.,  1.,  1.,  1.,  0., -1.,
         -1.,  1.,  0.]], device='cuda:0')
prediction
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0', grad_fn=<IndexPutBackward0>)
tensor_state_value[:, 1:] != 0
tensor([[ True, False,  True,  True,  True,  True, False,  True,  True,  True,
          True, False,  True,  True,  True, False]], device='cuda:0')
"""