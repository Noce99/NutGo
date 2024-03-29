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

    def play_a_match(self, agent1, agent2, wait=False, verbose=True, visualizer=True, red_starting=True):
        if visualizer:
            vi = Visualizer(self.board_size)
        else:
            vi = None
        start_state = HexNode(first_player=red_starting,
                              state_value=[[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                              default_policy=None,
                              probability_of_random_move=0)
        state = start_state
        while True:
            r, c, probability, best_child = agent1.get_move(vi, state, state.is_first_player())
            state = best_child

            if state.final_state:
                if state.winner_first_player:
                    if verbose:
                        print("First Player Won!", end=" ")
                    return True
                else:
                    if verbose:
                        print("Second Player Won!", end=" ")
                    return False
            if visualizer:
                vi.print_board(state.state_value, state.first_player)
                if probability is not None:
                    vi.print_probability(probability)
                if wait:
                    vi.wait_until_click()
            r, c, probability, best_child = agent2.get_move(vi, state, state.is_first_player())
            state = best_child

            if state.final_state:
                if state.winner_first_player:
                    if verbose:
                        print("First Player Won!", end=" ")
                    return True
                else:
                    if verbose:
                        print("Second Player Won!", end=" ")
                    return False
            if visualizer:
                vi.print_board(state.state_value, state.first_player)
                if probability is not None:
                    vi.print_probability(probability, offset=30)
                if wait:
                    vi.wait_until_click()


if __name__ == "__main__":
    dlr_agent_2_2 = HexAgent(board_size=7, batch_size=32, learning_rate=3, max_num_of_data=5000)
    dlr_agent_2_2.load_weight("7x7_Nut_v_2_2")
    dlr_agent_2_0 = HexAgent(board_size=7, batch_size=32, learning_rate=3, max_num_of_data=5000)
    dlr_agent_2_0.load_weight("7x7_Nut_v_2_0")
    #dlr_agent.evaluate_model()
    #dlr_agent.save_dataset()
    #dlr_agent.load_dataset("7x7_2023_04_07_23_09_42")
    #dlr_agent.single_training(300)
    #dlr_agent.trainer.plot_loss()
    #dlr_agent.evaluate_with_random_model(matches=100)
    #dlr_agent.train_while_playing(epochs=100, time_limit=20, simulations_limit=1500, num_of_games_before_evaluation=10,
    #                              prob_of_random_move=0.5, do_evaluation=False, save_dataset=True, max_num_of_games=None)
    # dlr_agent.evaluate_model()
    #dlr_agent.manually_evaluation()
    #dlr_agent.save_weight()

    a_game = HexGame(7)
    human = HumanAget()
    win_2_0 = 0
    win_2_2 = 0
    for i in range(50):
        if a_game.play_a_match(dlr_agent_2_0, dlr_agent_2_2, wait=False, verbose=False):
            win_2_0 += 1
        else:
            win_2_2 += 1
    print(f"2_0 = {win_2_0}, 2_2 = {win_2_2}")
    win_2_0 = 0
    win_2_2 = 0
    for i in range(50):
        if a_game.play_a_match(dlr_agent_2_2, dlr_agent_2_0, wait=False, verbose=False):
            win_2_2 += 1
        else:
            win_2_0 += 1
    print(f"2_0 = {win_2_0}, 2_2 = {win_2_2}")
    #a_game.play_a_match(human, MCTSAget(time=1, max_num_of_simulations=2000), wait=0)
    #vi = Visualizer(7)
    #vi.show_dataset(dlr_agent.dataset)
    #vi.visual_evaluation(dlr_agent.dataset, dlr_agent.trainer)

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
