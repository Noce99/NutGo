import copy
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from Neural_Networks import HexLinearModel, Trainer, HexConvolutionalModel
from Hex import HexNode
from NutGo.Agent_Random import RandomAgent
from NutGo.HexDisplay import Visualizer
from monte_carlo_tree_search import MCTS
from Agent import Agent


class HexAgent(Agent):

    def __init__(self, board_size, batch_size, learning_rate, max_num_of_data):
        self.board_size = board_size
        self.dataset = []

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_num_of_data = max_num_of_data
        # self.model = HexLinearModel(board_size)
        self.model = HexConvolutionalModel(board_size)
        self.trainer = Trainer(self.batch_size, self.learning_rate, self.model)
        self.bosses = [100]
        self.bosses_time_limit = 15

    def train_while_playing(self, epochs, time_limit, simulations_limit):
        # self.training_session(epochs, time_limit, simulations_limit, num_of_games=30, prob_of_random_move=1.0)
        # self.save_dataset()
        #self.evaluate_with_random_model(matches=1000)
        start = time.time()
        x_time = [0]
        y_evaluations = [self.evaluate_model()]
        num_of_games = 0
        while True:
            self.training_session(epochs, time_limit, simulations_limit, num_of_games=10, prob_of_random_move=0.3)
            score = self.evaluate_model()
            num_of_games += 10
            if score > max(y_evaluations):
                self.save_weight()
            y_evaluations.append(score)
            now = time.time() - start
            x_time.append(now)
            plt.plot(x_time, y_evaluations)
            plt.savefig(f'evaluations/eval_{int(now)}.png')
            print(f"DONE {num_of_games} in {now} seconds!")

    def training_session(self, epochs, time_limit, simulations_limit, num_of_games, prob_of_random_move):
        red_was_the_winner = 0
        blue_was_the_winner = 0
        print(f"Start training for {num_of_games} games with {prob_of_random_move} prob_of_random_move.")
        for nog in range(num_of_games):
            start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                                  self.trainer, prob_of_random_move)
            state = start_state
            while True:
                r, c, probability = self.get_a_move_from_mcts(state.state_value, first_player=True,
                                                              prob_of_random_move=1.0,
                                                              default_policy=self.get_a_move_from_nn,
                                                              time_limit=time_limit,
                                                              max_num_of_simulations=simulations_limit)
                new_state = HexNode(first_player=False,
                                    state_value=copy.deepcopy(state.state_value),
                                    added_piece=(r, c))
                new_state.add_piece(r, c, piece_is_of_first_player=True)
                self.add_data_to_dataset(state.state_value, state.is_first_player(), probability,
                                         check_already_there=True)
                state = new_state
                if state.final_state:
                    if state.winner_first_player:
                        print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
                r, c, probability = self.get_a_move_from_mcts(state.state_value, first_player=False,
                                                              prob_of_random_move=prob_of_random_move,
                                                              default_policy=self.get_a_move_from_nn,
                                                              time_limit=time_limit,
                                                              max_num_of_simulations=simulations_limit)
                new_state = HexNode(first_player=True,
                                    state_value=copy.deepcopy(state.state_value),
                                    added_piece=(r, c))
                new_state.add_piece(r, c, piece_is_of_first_player=False)
                self.add_data_to_dataset(state.state_value, state.is_first_player(), probability,
                                         check_already_there=True)
                state = new_state
                if state.final_state:
                    if state.winner_first_player:
                        print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
            self.just_train(epochs)
            print(f"{nog + 1}/{num_of_games} game finished {len(self.dataset)} data.")
        print(f"Red/Blue wins -> {red_was_the_winner}/{blue_was_the_winner}")

    def just_train(self, epochs, dataset=None):
        if dataset is None:
            dataset = self.dataset
        batches = [self.get_a_batch_of_data(dataset=dataset) for _ in range(4)]
        self.trainer.train(epochs, batches)

    def single_training(self, epochs):
        training = self.dataset[:int(0.9*len(self.dataset))]
        test = self.dataset[int(0.9*len(self.dataset)):]
        for i in range(epochs // 10):
            self.just_train(10, training)
        print("Training:")
        self.plot_accuracy()
        print("Test:")
        self.plot_accuracy(dataset=test)

    def add_data_to_dataset(self, state_value, first_player, probability, check_already_there=False):
        data_x = from_state_value_to_1d_tensor(state_value, first_player)
        data_y = probability
        if check_already_there:
            for i in range(len(self.dataset)):
                if torch.equal(self.dataset[i][0], data_x):
                    del self.dataset[i]
                    break
        if torch.sum(data_y) > 0.1:
            self.dataset.append((data_x, data_y))
            if len(self.dataset) > self.max_num_of_data:
                del self.dataset[0]

    def save_dataset(self):
        if len(self.dataset) == 0:
            print("Impossible to save an empty dataset!")
            exit()
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets")
        current_time = datetime.now()
        data_x = self.dataset[0][0][None, :]
        data_y = self.dataset[0][1][None, :]
        for d_x, d_y in self.dataset[1:]:
            data_x = torch.cat((data_x, d_x[None, :]), 0)
            data_y = torch.cat((data_y, d_y[None, :]), 0)
        torch.save(data_x, f"./datasets/{self.board_size}x{self.board_size}_"
                           f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.X")
        torch.save(data_y, f"./datasets/{self.board_size}x{self.board_size}_"
                           f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.Y")

    def load_dataset(self, dataset_name):
        if os.path.exists("./datasets/" + dataset_name + ".X") and os.path.exists("./datasets/" + dataset_name + ".Y"):
            data_x = torch.load("./datasets/" + dataset_name + ".X")
            data_y = torch.load("./datasets/" + dataset_name + ".Y")
            print(f"X_shape: {data_x.shape}")
            print(f"Y_shape: {data_y.shape}")
            if data_x.shape[0] != data_y.shape[0]:
                print(f"[{'./datasets/' + dataset_name + '.X'}] and [{'./datasets/' + dataset_name + '.Y'}]"
                      f" have different sizes!")
            num_of_data = data_x.shape[0]
            self.dataset = []
            for i in range(num_of_data):
                self.dataset.append((data_x[i, :], data_y[i, :]))
            self.print_how_much_data()
            return None
        elif not os.path.exists("./datasets/" + dataset_name + ".X"):
            print(f"[{'./datasets/' + dataset_name + '.X'}] doesn't exists!")
        elif not os.path.exists("./datasets/" + dataset_name + ".Y"):
            print(f"[{'./datasets/' + dataset_name + '.Y'}] doesn't exists!")
        exit()

    def get_a_batch_of_data(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        random_index = [random.randrange(0, len(dataset)) for _ in range(self.batch_size)]
        data_x = dataset[random_index[0]][0]
        data_y = dataset[random_index[0]][1]
        for i in random_index[1:]:
            data_x = torch.cat((data_x, dataset[i][0]), 0)
            data_y = torch.cat((data_y, dataset[i][1]), 0)
        return data_x, data_y

    def save_weight(self):
        if not os.path.exists("./weights"):
            os.mkdir("./weights")
        current_time = datetime.now()
        torch.save(self.model.state_dict(), f"./weights/{self.board_size}x{self.board_size}_"
                                            f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.w")

    def load_weight(self, weight_name):
        if os.path.exists("./weights/" + weight_name):
            self.model.load_state_dict(torch.load("./weights/" + weight_name))
            # self.model.eval()
        else:
            print(f"[{'./weights/' + weight_name}] doesn't exists!")
            exit()

    def get_a_move_from_nn(self, state_value, first_player):
        tensor_state_value = from_state_value_to_1d_tensor(state_value, first_player)
        prediction = self.trainer.predict(tensor_state_value)[0]
        random_float = random.random()
        actual_sum = 0
        """
        Get move with probability!
        
        selected_index = -1
        for i in range(prediction.shape[0]):
            actual_sum += prediction[i]
            if random_float <= actual_sum:
                selected_index = i
                break
        """
        selected_index = torch.argmax(prediction)
        if selected_index == -1:
            print(f"Actual Sum: {actual_sum}")
            print(f"Random Float: {random_float}")
            print("Prediction:")
            print(prediction)
            print("State Value:")
            print(state_value)
            print("Tensor state value:")
            print(tensor_state_value)
            assert "Impossible to arrive there!"
            return None
        else:
            r = selected_index // self.board_size
            c = selected_index - r * self.board_size
            return r, c, prediction

    def get_a_move_from_mcts(self, state_value, first_player, prob_of_random_move, default_policy,
                             time_limit, max_num_of_simulations=-1):
        actual_s = HexNode(first_player, state_value, None, default_policy, prob_of_random_move)
        my_mcts = MCTS(actual_s, time_limit, max_num_of_simulations)
        result = from_result_to_1d_tensor(len(state_value), my_mcts.explore(verbose=False))
        max_i = torch.argmax(result[0])
        ######################################################
        # Provo con result con tutti zero e un solo uno
        # new_result = torch.zeros_like(result)
        # new_result[0][max_i] = 1
        # result = new_result
        ######################################################
        r = max_i // self.board_size
        c = max_i - (self.board_size * r)
        return r, c, result

    def get_move(self, _, state_value, first_player):
        return self.get_a_move_from_nn(state_value, first_player)

    def print_how_much_data(self):
        print(f"I have {len(self.dataset)} data!")

    def plot_accuracy(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        correct = 0
        n = 0
        for x, y in dataset:
            max_y = torch.argmax(y[0])
            predicted_tensor = self.trainer.predict(x)
            max_predicted_y = torch.argmax(predicted_tensor)
            if max_y == max_predicted_y:
                correct += 1
            n += 1
        print(f"Accuracy: {correct / n}")

    def play_a_game_with_mcts(self, mcts_is_first_player, time_limit, simulations_limit=-1, visualize=False):
        """

        :param visualize:
        :param mcts_is_first_player:
        :param time_limit:
        :param simulations_limit:
        :return: Return True if NN won and False if mcts won
        """
        if visualize:
            vi = Visualizer(self.board_size)
        start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                              self.trainer, 0.0)
        state = start_state
        if visualize:
            vi.print_board(state.state_value, None)
        while True:
            if mcts_is_first_player:
                r, c, _ = self.get_a_move_from_mcts(state.state_value, first_player=True,
                                                    prob_of_random_move=1.0,
                                                    default_policy=None,
                                                    time_limit=time_limit,
                                                    max_num_of_simulations=simulations_limit)
                state = HexNode(first_player=False,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=True)
                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return False
                r, c, _ = self.get_a_move_from_nn(state.state_value, first_player=False)
                state = HexNode(first_player=True,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=False)

                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return True
            else:
                r, c, _ = self.get_a_move_from_nn(state.state_value, first_player=True)
                state = HexNode(first_player=False,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=True)

                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return True
                r, c, _ = self.get_a_move_from_mcts(state.state_value, first_player=False,
                                                    prob_of_random_move=1.0,
                                                    default_policy=None,
                                                    time_limit=time_limit,
                                                    max_num_of_simulations=simulations_limit)
                state = HexNode(first_player=True,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=False)
                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return False
            if visualize:
                vi.print_board(state.state_value, None)
                time.sleep(1)

    def evaluate_model(self):
        score = 0
        print("EVALUATION:")
        for i, bos in enumerate(self.bosses):
            bos_win = 0
            nn_win = 0
            # nn first_player
            for game in range(50):
                if self.play_a_game_with_mcts(mcts_is_first_player=False, time_limit=self.bosses_time_limit,
                                              simulations_limit=bos, visualize=False):
                    nn_win += 1
                    # print("NN won with red!")
                else:
                    bos_win += 1
                    # print("Bos won with blue!")
            # bos first_player
            for game in range(50):
                if self.play_a_game_with_mcts(mcts_is_first_player=True, time_limit=self.bosses_time_limit,
                                              simulations_limit=bos, visualize=False):
                    nn_win += 1
                    # print("NN won with blue!")
                else:
                    bos_win += 1
                    # print("Bos won with red!")
            print(f"NN vs bos_{i} [{bos}]: {nn_win}/{bos_win} wins! [{nn_win/(nn_win+bos_win) * 100} %]")
            return nn_win/(nn_win+bos_win) * 100
            if nn_win >= bos_win:
                score += 1
            else:
                break
        print(f"SCORE={score}")

    def play_a_game_with_random_model(self, random_is_first_player):
        start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                              self.trainer, 0.0)
        state = start_state
        ra = RandomAgent()
        while True:
            if random_is_first_player:
                r, c, _ = ra.get_move(vi=None, state_value=state.state_value)
                state = HexNode(first_player=False,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=True)
                if state.final_state:
                    return False
                r, c, _ = self.get_a_move_from_nn(state.state_value, first_player=False)
                state = HexNode(first_player=True,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=False)

                if state.final_state:
                    return True
            else:
                r, c, _ = self.get_a_move_from_nn(state.state_value, first_player=True)
                state = HexNode(first_player=False,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=True)

                if state.final_state:
                    return True
                r, c, _ = ra.get_move(vi=None, state_value=state.state_value)
                state = HexNode(first_player=True,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=False)
                if state.final_state:
                    return False

    def evaluate_with_random_model(self, matches=100):
        nn_won = 0
        random_won = 0
        for match in range(matches):
            random_first_player = True
            if match > matches // 2:
                random_first_player = False
            if self.play_a_game_with_random_model(random_first_player):
                nn_won += 1
            else:
                random_won += 1
        print(f"NN Wins / Random Wins -> {nn_won}/{random_won} [{nn_won/(random_won + nn_won) * 100:.2f} %]")

    def manually_evaluation(self):
        vi = Visualizer(self.board_size)
        for i in range(len(self.dataset)):
            state = self.dataset[i][0][0]
            result = self.dataset[i][1][0]
            hex_state = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
            r = 0
            c = 0
            for ii, element in enumerate(state[1:]):
                if element == 1:
                    hex_state[r][c] = (True, None)
                elif element == -1:
                    hex_state[r][c] = (False, None)
                c += 1
                if (ii + 1) % self.board_size == 0:
                    r += 1
                    c = 0
            if state[0] == 1:
                vi.print_board(hex_state, True)
            else:
                vi.print_board(hex_state, False)
            vi.print_probability(self.dataset[i][1][0])
            vi.print_probability(self.trainer.predict(self.dataset[i][0]), offset=30)
            vi.wait_until_click()


def from_state_value_to_1d_tensor(my_state_value, first_player):
    """
    First of all I simplify the representation removing the knowledge, -1 for second player pieces,
    0 for empty cell and 1 for first player pieces, and finally I flat the array.
    """
    new_state_value = [[0 for _ in range(len(my_state_value))] for _ in range(len(my_state_value))]
    for r, raw in enumerate(my_state_value):
        for c, cell in enumerate(raw):
            if cell is not None:
                # Cell is not empty
                if cell[0] is True:
                    new_state_value[r][c] = 1
                else:
                    new_state_value[r][c] = -1
    new_state_value = torch.Tensor(new_state_value)
    if first_player:
        new_state_value = torch.cat((torch.Tensor([1]), torch.flatten(new_state_value)))
    else:
        new_state_value = torch.cat((torch.Tensor([-1]), torch.flatten(new_state_value)))
    """
    print("Old State Value:")
    for raw in my_state_value:
        print(raw)
    print("New State Value")
    print(new_state_value)
    """
    new_state_value = new_state_value[None, :]
    return new_state_value


def from_result_to_1d_tensor(table_size, my_result):
    new_result = torch.zeros(table_size, table_size)
    for r, value in my_result:
        position = r.added_piece
        new_result[position[0], position[1]] = value
    """
    print("Old Result:")
    for node, value in my_result:
        print(node.added_piece, value)
    print("New Result")
    print(new_result)
    """
    new_result = torch.flatten(new_result)
    tensor_sum = torch.sum(new_result)
    if tensor_sum == 0:
        tensor_sum = 1
    new_result = new_result / tensor_sum
    new_result = new_result[None, :]
    return new_result
