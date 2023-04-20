import time

import torch
from matplotlib import pyplot as plt

from Agent_DRL import HexAgent
from Game_Player import HexGame


class TOPP:

    def __init__(self, board_size, total_episodes=200, m=5, g=25, batch_size=32, learning_rate=3, max_num_of_data=10000,
                 optimizer="SGD", directory="./demo", epochs=500, time_limit=10, simulations_limit=2000,
                 prob_of_random_move=0.8, hidden_layers=0, neuron_per_layers=64, activation_function="RELU"):
        self.board_size = board_size
        self.total_episodes = total_episodes
        self.m = m
        self.g = g
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_num_of_data = max_num_of_data
        self.optimizer = optimizer
        self.directory = directory
        self.epochs = epochs
        self.time_limit = time_limit
        self.simulations_limit = simulations_limit
        self.prob_of_random_move = prob_of_random_move
        self.hidden_layers = hidden_layers
        self.neuron_per_layers = neuron_per_layers
        self.activation_function = activation_function

    def train(self):
        dlr_agent = HexAgent(board_size=self.board_size, batch_size=self.batch_size, learning_rate=self.learning_rate,
                             max_num_of_data=self.max_num_of_data, optimizer=self.optimizer,
                             hidden_layers=self.hidden_layers, neuron_per_layers=self.neuron_per_layers,
                             activation_function=self.activation_function)
        dlr_agent.save_weight(folder=self.directory, name="player_0")
        for i in range(1, self.m):
            dlr_agent.train_while_playing(epochs=self.epochs, time_limit=self.time_limit,
                                          simulations_limit=self.simulations_limit,
                                          num_of_games_before_evaluation=self.total_episodes // self.m,
                                          prob_of_random_move=self.prob_of_random_move, do_evaluation=False,
                                          save_dataset=False, max_num_of_games=self.total_episodes // self.m)
            # dlr_agent.evaluate_model()
            time.sleep(0.1)
            dlr_agent.save_weight(folder=self.directory, name=f"player_{i}")

    def play_a_match(self, agent_1, agent_2, visualizer=False):
        half_g = self.g // 2
        agent_1_wins_fp = 0
        agent_1_wins_sp = 0
        agent_2_wins_fp = 0
        agent_2_wins_sp = 0
        for i in range(half_g):
            a_game = HexGame(self.board_size)
            result = a_game.play_a_match(agent_1, agent_2, wait=visualizer, verbose=False, visualizer=visualizer)
            if result:
                agent_1_wins_fp += 1
            else:
                agent_2_wins_sp += 1
        for i in range(half_g):
            a_game = HexGame(self.board_size)
            result = a_game.play_a_match(agent_2, agent_1, wait=visualizer, verbose=False, visualizer=visualizer)
            if result:
                agent_2_wins_fp += 1
            else:
                agent_1_wins_sp += 1
        return agent_1_wins_fp, agent_1_wins_sp, agent_2_wins_fp, agent_2_wins_sp

    def play(self, visualizer=False):
        players = [HexAgent(board_size=self.board_size, batch_size=self.batch_size, learning_rate=self.learning_rate,
                            max_num_of_data=self.max_num_of_data, optimizer=self.optimizer,
                            hidden_layers=self.hidden_layers, neuron_per_layers=self.neuron_per_layers,
                            activation_function=self.activation_function)
                   for _ in range(self.m)]
        num_of_wins = [[0, 0] for _ in range(self.m)]
        for i in range(self.m):
            players[i].load_weight(f"{self.board_size}x{self.board_size}_player_{i}", folder=self.directory)
        for i in range(self.m - 1):
            for j in range(i + 1, self.m):
                agent_1_wins_fp, agent_1_wins_sp, agent_2_wins_fp, agent_2_wins_sp =\
                    self.play_a_match(players[i], players[j], visualizer)
                num_of_wins[i][0] += agent_1_wins_fp
                num_of_wins[i][1] += agent_1_wins_sp
                num_of_wins[j][0] += agent_2_wins_fp
                num_of_wins[j][1] += agent_2_wins_sp
        for i in range(self.m):
            # print(f"player_{i} won {num_of_wins[i][0]+num_of_wins[i][1]} [{num_of_wins[i]}] times.")
            print(f"player_{i} won {num_of_wins[i][0]+num_of_wins[i][1]} times.")

    def train_solo(self):
        players = [HexAgent(board_size=self.board_size, batch_size=self.batch_size, learning_rate=self.learning_rate,
                            max_num_of_data=self.max_num_of_data, optimizer=self.optimizer,
                            hidden_layers=self.hidden_layers, neuron_per_layers=self.neuron_per_layers,
                            activation_function=self.activation_function)
                   for _ in range(self.m)]
        for i in range(1, self.m):
            players[i].load_dataset(f"{self.board_size}x{self.board_size}_player_{i}")

        train_epochs = [0, 10, 100, 500, 1000]

        for i in range(1, self.m):
            players[i].just_train(train_epochs[i])
            plt.plot(list(range(len(players[i].trainer.train_history))), players[i].trainer.train_history,
                     label=f"player {i}")
        plt.legend()
        plt.show()
        for i in range(self.m):
            players[i].save_weight(folder=self.directory, name=f"player_{i}")

    def show_a_match(self, id_player_1, id_player_2):
        players = [
            HexAgent(board_size=self.board_size, batch_size=self.batch_size, learning_rate=self.learning_rate,
                     max_num_of_data=self.max_num_of_data, optimizer=self.optimizer,
                     hidden_layers=self.hidden_layers, neuron_per_layers=self.neuron_per_layers,
                     activation_function=self.activation_function)
            for _ in range(self.m)]
        players[id_player_1].load_weight(f"{self.board_size}x{self.board_size}_player_{id_player_1}",
                                         folder=self.directory)
        players[id_player_2].load_weight(f"{self.board_size}x{self.board_size}_player_{id_player_2}",
                                         folder=self.directory)
        a_game = HexGame(self.board_size)
        result = a_game.play_a_match(players[id_player_1], players[id_player_2], wait=True, verbose=False,
                                     visualizer=False)
        if result:
            print("Player 1 won!")
        else:
            print("Player 2 won!")


if __name__ == "__main__":
    tournament = TOPP(board_size=5, total_episodes=20, m=5, g=25, batch_size=32, learning_rate=3,
                      max_num_of_data=10000, optimizer="SGD", directory="./demo",  # "./long_trained_TOPP"
                      epochs=600, time_limit=10, simulations_limit=2000, prob_of_random_move=1.0, hidden_layers=0,
                      neuron_per_layers=64, activation_function="RELU")
    tournament.train()
    tournament.play(visualizer=False)
