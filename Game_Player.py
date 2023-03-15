import torch

from HexDisplay import Visualizer
from Agent_DRL import HexAgent, from_result_to_1d_tensor
from Hex import HexNode
from Human_Player import HumanAget
from monte_carlo_tree_search import MCTS
from Agent_MCTS import MCTSAget


class HexGame:
    def __init__(self, board_size):
        self.board_size = board_size

    def play_a_match(self, agent1, agent2):
        vi = Visualizer(self.board_size)
        start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)], None)
        state = start_state
        probability = None
        while True:
            r, c, probability = agent1.get_move(vi, state.state_value, state.first_player)
            state = HexNode(first_player=not state.is_first_player(),
                            state_value=state.state_value[:],
                            added_piece=(r, c))
            state.add_piece(r, c, not state.is_first_player())
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
            r, c, probability = agent2.get_move(vi, state.state_value, state.first_player)
            state = HexNode(first_player=not state.is_first_player(),
                            state_value=state.state_value[:],
                            added_piece=(r, c))
            state.add_piece(r, c, not state.is_first_player())
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
    dlr_agent = HexAgent(board_size=4, batch_size=64, learning_rate=0.1, max_num_of_batches=30)
    #dlr_agent.load_dataset("4x4_validation")
    #dlr_agent.train_while_playing(epochs=100, time_limit=5, simulations_limit=4000, num_of_games=200,
    #                              prob_of_random_move=0.7)
    #dlr_agent.save_dataset()
    #dlr_agent.load_dataset("4x4_2023_03_13_11_47_49")
    #dlr_agent.trainer.train(1000, dlr_agent.dataset)
    #dlr_agent.trainer.plot_loss()
    #dlr_agent.plot_accuracy()
    #dlr_agent.save_weight()

    dlr_agent.load_weight("it_s_something.w")

    a_game = HexGame(4)
    human = HumanAget()
    a_game.play_a_match(human, dlr_agent, )
    # vi = Visualizer(4)
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