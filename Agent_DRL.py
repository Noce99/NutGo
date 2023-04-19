import os
import random
import time
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
from Neural_Networks import Trainer, HexConvolutionalModel
from Hex import HexNode
from HexDisplay import Visualizer
from monte_carlo_tree_search import MCTS
from Agent import Agent
from Utils import my_argmax


class HexAgent(Agent):

    def __init__(self, board_size, batch_size, learning_rate, max_num_of_data, optimizer="SGD",
                 hidden_layers=0, neuron_per_layers=64, activation_function="RELU"):
        self.board_size = board_size
        self.dataset = []

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_num_of_data = max_num_of_data
        self.model = HexConvolutionalModel(board_size, hidden_layers=hidden_layers, neuron_per_layers=neuron_per_layers,
                                           activation_function=activation_function)
        self.trainer = Trainer(self.batch_size, self.learning_rate, self.model, optimizer)

        self.bosses = [100]
        self.bosses_time_limit = 15

    def train_while_playing(self, epochs, time_limit, simulations_limit, prob_of_random_move,
                            num_of_games_before_evaluation, do_evaluation=True, save_dataset=True,
                            max_num_of_games=None):
        start = time.time()
        x_time = [0]
        y_evaluations = []
        if do_evaluation:
            y_evaluations.append(self.evaluate_model())
        num_of_games = 0
        while True:
            self.training_session(epochs, time_limit, simulations_limit, num_of_games=num_of_games_before_evaluation,
                                  prob_of_random_move=prob_of_random_move)
            num_of_games += num_of_games_before_evaluation
            now = time.time() - start
            if do_evaluation:
                score = self.evaluate_model()
                y_evaluations.append(score)
                x_time.append(now)
                plt.plot(x_time, y_evaluations)
                plt.savefig(f'evaluations/eval_{int(now)}.png')
            self.save_weight()
            if save_dataset:
                self.save_dataset()
            print(f"DONE {num_of_games} in {now} seconds!")
            if max_num_of_games is not None:
                if max_num_of_games <= num_of_games:
                    break

    def training_session(self, epochs, time_limit, simulations_limit, num_of_games, prob_of_random_move):
        red_was_the_winner = 0
        blue_was_the_winner = 0
        print(f"Start training for {num_of_games} games with {prob_of_random_move} prob_of_random_move.")
        time.sleep(0.1)
        training_tqdm = tqdm(range(0, num_of_games), unit=" Game", desc="Training")
        for nog in training_tqdm:
            start_state = HexNode(first_player=True,
                                  state_value=[[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                                  default_policy=self.get_a_move_from_nn,
                                  probability_of_random_move=prob_of_random_move)
            state = start_state
            while True:

                r, c, probability, best_child = self.get_a_move_from_mcts(state,
                                                                          time_limit=time_limit,
                                                                          max_num_of_simulations=simulations_limit)
                self.add_data_to_dataset(state, state.is_first_player(), probability)
                state = best_child

                if state.final_state:
                    if state.winner_first_player:
                        # print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        # print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
                r, c, probability, best_child = self.get_a_move_from_mcts(state,
                                                                          time_limit=time_limit,
                                                                          max_num_of_simulations=simulations_limit)
                self.add_data_to_dataset(state, state.is_first_player(), probability)
                state = best_child

                if state.final_state:
                    if state.winner_first_player:
                        # print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        # print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
            self.just_train(epochs)
            # print(f"{nog + 1}/{num_of_games} game finished {len(self.dataset)}} data.")
        # print(f"Red/Blue wins -> {red_was_the_winner}/{blue_was_the_winner}")

    def just_train(self, epochs, datasets=None):
        if datasets is None:
            datasets = self.dataset
        batches = [self.get_a_batch_of_data(dataset=datasets) for _ in range(4)]
        self.trainer.train(epochs, batches)

    """
    def single_training(self, epochs, first_player):
        dataset = self.dataset
        trainer = self.trainer
        train = dataset[:int(0.9 * len(dataset))]
        test = dataset[int(0.9 * len(dataset)):]
        for i in range(epochs // 10):
            self.just_train(epochs=10, datasets=train)
        print("Train:")
        self.plot_accuracy(dataset=train, trainer=trainer)
        print("Test:")
        self.plot_accuracy(dataset=test, trainer=trainer)
    """

    def add_data_to_dataset(self, state, first_player, probability):
        data_x = from_state_value_to_tensor(state, first_player)
        data_y = probability
        dataset = self.dataset
        if torch.sum(data_y) > 0.1:
            dataset.append((data_x, data_y))
            if len(dataset) > self.max_num_of_data:
                del dataset[0]

    def save_dataset(self):
        def save_single_dataset(dataset, board_size, first_player):
            if len(dataset) == 0:
                print("Impossible to save an empty dataset!")
                exit()
            if not os.path.exists("./datasets"):
                os.mkdir("./datasets")
            current_time = datetime.now()
            data_x = dataset[0][0][None, :]
            data_y = dataset[0][1][None, :]
            for d_x, d_y in dataset[1:]:
                data_x = torch.cat((data_x, d_x[None, :]), 0)
                data_y = torch.cat((data_y, d_y[None, :]), 0)
            torch.save(data_x, f"./datasets/{board_size}x{board_size}_"
                               f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.X")
            torch.save(data_y, f"./datasets/{board_size}x{board_size}_"
                               f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.Y")

        save_single_dataset(self.dataset, self.board_size, True)

    def load_dataset(self, out_dataset_name):
        def load_single_dataset(dataset_name):
            dataset = None
            if os.path.exists("./datasets/" + dataset_name + ".X") and os.path.exists(
                    "./datasets/" + dataset_name + ".Y"):
                data_x = torch.load("./datasets/" + dataset_name + ".X")
                data_y = torch.load("./datasets/" + dataset_name + ".Y")
                print(f"X_shape: {data_x.shape}")
                print(f"Y_shape: {data_y.shape}")
                if data_x.shape[0] != data_y.shape[0]:
                    print(f"[{'./datasets/' + dataset_name + '.X'}] and [{'./datasets/' + dataset_name + '.Y'}]"
                          f" have different sizes!")
                num_of_data = data_x.shape[0]
                dataset = []
                for i in range(num_of_data):
                    dataset.append((data_x[i, :], data_y[i, :]))
            elif not os.path.exists("./datasets/" + dataset_name + ".X"):
                print(f"[{'./datasets/' + dataset_name + '.X'}] doesn't exists!")
            elif not os.path.exists("./datasets/" + dataset_name + ".Y"):
                print(f"[{'./datasets/' + dataset_name + '.Y'}] doesn't exists!")
            return dataset

        self.dataset = load_single_dataset(f"{out_dataset_name}")
        self.print_how_much_data()

    def get_a_batch_of_data(self, dataset=None):
        random_index = [random.randrange(0, len(dataset)) for _ in range(self.batch_size)]
        data_x = (dataset[random_index[0]][0])[None, :]
        data_y = (dataset[random_index[0]][1])[None, :]
        for i in random_index[1:]:
            data_x = torch.cat((data_x, dataset[i][0][None, :]), 0)
            data_y = torch.cat((data_y, dataset[i][1][None, :]), 0)
        return data_x, data_y

    def save_weight(self, folder="./weights", name=None):
        if not os.path.exists(folder):
            os.mkdir(folder)
        if name is None:
            current_time = datetime.now()
            name = current_time.strftime('%Y_%m_%d_%H_%M_%S')
        torch.save(self.model.state_dict(), f"{folder}/{self.board_size}x{self.board_size}_{name}.w")

    def load_weight(self, weight_name, folder="./weights"):
        if os.path.exists(folder + "/" + weight_name + ".w"):
            self.model.load_state_dict(torch.load(folder + "/" + weight_name + ".w"))
        else:
            print(f"{folder + '/' + weight_name + '.w'} doesn't exists!")
            exit()

    def get_a_move_from_nn(self, state, first_player, check_obvious_moves=False):
        tensor_state_value = from_state_value_to_tensor(state, first_player)
        if check_obvious_moves:
            obv_r, obv_c = find_obvious_moves(tensor_state_value, first_player)
            if obv_r is not None:
                # print(f"Obviously: [{obv_r}, {obv_c}]")
                return obv_r, obv_c, None
        prediction = self.trainer.predict(tensor_state_value)

        selected_index = my_argmax(prediction)
        return selected_index[0], selected_index[1], prediction

        my_prediction = torch.clone(prediction)

        best_moves = []
        best_moves_values = []
        for i in range(3):
            selected_index = my_argmax(my_prediction)
            if float(my_prediction[selected_index[0], selected_index[1]]) == 0:
                break
            best_moves.append(selected_index)
            best_moves_values.append(float(my_prediction[selected_index[0], selected_index[1]]))
            my_prediction[selected_index[0], selected_index[1]] = 0
        """
        print("@" * 120)
        print(prediction)
        print("#" * 120)
        print(my_prediction)
        print("@"*120)
        """
        the_sum = sum(best_moves_values)
        for i in range(len(best_moves)):
            best_moves_values[i] = best_moves_values[i] / the_sum
        random_number = random.uniform(0, 1)
        the_sum = 0
        for i in range(len(best_moves)):
            the_sum += best_moves_values[i]
            if random_number < the_sum:
                return best_moves[i][0], best_moves[i][1], prediction

    def get_a_move_from_mcts(self, state, time_limit, max_num_of_simulations=-1):
        my_mcts = MCTS(state, time_limit, max_num_of_simulations)
        result = my_mcts.explore(verbose=False)
        best_value = 0
        best_child = None
        for child, value in result:
            if value > best_value:
                best_value = value
                best_child = child
        tensor_result = from_result_to_tensor(self.board_size, result)
        selected_index = my_argmax(tensor_result)

        if best_child is None:
            print(len(result))
            print(result)
            print(state)
        return selected_index[0], selected_index[1], tensor_result, best_child

    def get_move(self, _, state, first_player):
        r, c, probability = self.get_a_move_from_nn(state, first_player, check_obvious_moves=False)
        best_child = get_child_selected_by_nn(r, c, state)
        if best_child is None:
            print(f"[{r}, {c}]")
        return r, c, probability, best_child

    def print_how_much_data(self):
        print(f"I have {len(self.dataset)} data!")

    """
    @staticmethod
    def plot_accuracy(dataset=None, trainer=None):
        correct = 0
        n = 0
        for x, y in dataset:
            max_y = torch.argmax(y)
            predicted_tensor = trainer.predict(x)
            max_predicted_y = torch.argmax(predicted_tensor)
            if max_y == max_predicted_y:
                correct += 1
            n += 1
        print(f"Accuracy: {correct / n}")
    """

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
        start_state = HexNode(first_player=True,
                              state_value=[[None for _ in range(self.board_size)] for _ in range(self.board_size)])
        state = start_state
        if visualize:
            vi.print_board(state.state_value, None)
        while True:
            if mcts_is_first_player:
                _, _, _, state = self.get_a_move_from_mcts(state,
                                                           time_limit=time_limit,
                                                           max_num_of_simulations=simulations_limit)
                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return False
                r, c, _ = self.get_a_move_from_nn(state, first_player=False)
                state = get_child_selected_by_nn(r, c, state)

                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return True
            else:
                r, c, _ = self.get_a_move_from_nn(state, first_player=False)
                state = get_child_selected_by_nn(r, c, state)

                if state.final_state:
                    if visualize:
                        vi.print_board(state.state_value, None)
                        time.sleep(2)
                    return True

                _, _, _, state = self.get_a_move_from_mcts(state,
                                                           time_limit=time_limit,
                                                           max_num_of_simulations=simulations_limit)
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
            bos_win_1 = 0
            nn_win_1 = 0
            bos_win_2 = 0
            nn_win_2 = 0
            # nn first_player
            nn_first_player = tqdm(range(50), unit=" Game", desc="Evaluation nn first player")
            for game in nn_first_player:
                if self.play_a_game_with_mcts(mcts_is_first_player=False, time_limit=self.bosses_time_limit,
                                              simulations_limit=bos, visualize=False):
                    nn_win_1 += 1
                    # print("NN won with red!")
                else:
                    bos_win_1 += 1
                    # print("Bos won with blue!")
            print(f"nn first player: {nn_win_1 / (nn_win_1 + bos_win_1) * 100} % wins!")
            time.sleep(0.1)
            # bos first_player
            nn_second_player = tqdm(range(50), unit=" Game", desc="Evaluation nn second player")
            for game in nn_second_player:
                if self.play_a_game_with_mcts(mcts_is_first_player=True, time_limit=self.bosses_time_limit,
                                              simulations_limit=bos, visualize=False):
                    nn_win_2 += 1
                    # print("NN won with blue!")
                else:
                    bos_win_2 += 1
                    # print("Bos won with red!")
            print(f"nn second player: {nn_win_2 / (nn_win_2 + bos_win_2) * 100} % wins!")

            nn_win = nn_win_1 + nn_win_2
            bos_win = bos_win_1 + bos_win_2
            print(f"NN vs bos_{i} [{bos}]: {nn_win}/{bos_win} wins! [{nn_win / (nn_win + bos_win) * 100} %]")
            return nn_win / (nn_win + bos_win) * 100
            if nn_win >= bos_win:
                score += 1
            else:
                break
        print(f"SCORE={score}")

    """
    def play_a_game_with_random_model(self, random_is_first_player):
        print("BROKEN!!!!!!!!!!")
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
                r, c, _ = self.get_a_move_from_nn(state, first_player=False)
                state = HexNode(first_player=True,
                                state_value=copy.deepcopy(state.state_value),
                                added_piece=(r, c))
                state.add_piece(r, c, piece_is_of_first_player=False)

                if state.final_state:
                    return True
            else:
                r, c, _ = self.get_a_move_from_nn(state, first_player=True)
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
        print(f"NN Wins / Random Wins -> {nn_won}/{random_won} [{nn_won / (random_won + nn_won) * 100:.2f} %]")

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
    """


def from_state_value_to_tensor(state, first_player):
    """
    First of all I simplify the representation removing the knowledge, -1 for second player pieces,
    0 for empty cell and 1 for first player pieces, and finally I flat the array.
    """
    board_size = len(state.state_value)
    pieces = torch.zeros(board_size, board_size)
    first_player_connection = torch.zeros(board_size, board_size)
    second_player_connection = torch.zeros(board_size, board_size)
    for r, raw in enumerate(state.state_value):
        for c, cell in enumerate(raw):
            if cell is not None:
                if cell[0] is True:
                    # first player peace there
                    if first_player:
                        pieces[r, c] = 1
                    else:
                        pieces[r, c] = -1
                    if cell[1] is True:
                        # Connected to the left (top) line
                        first_player_connection[r, c] = 1
                    elif cell[1] is False:
                        # Connected to the right (bottom) line
                        first_player_connection[r, c] = -1
                elif cell[0] is False:
                    # second player peace there
                    if first_player:
                        pieces[r, c] = -1
                    else:
                        pieces[r, c] = 1
                    if cell[1] is True:
                        # Connected to the left (top) line
                        second_player_connection[r, c] = 1
                    elif cell[1] is False:
                        # Connected to the right (bottom) line
                        second_player_connection[r, c] = -1

    if first_player:
        tensor_state_value = torch.cat((first_player_connection[None, :], second_player_connection[None, :],
                                        pieces[None, :]), dim=0)
    else:
        tensor_state_value = torch.cat((second_player_connection[None, :], first_player_connection[None, :],
                                        pieces[None, :]), dim=0)
    return tensor_state_value


def from_result_to_tensor(table_size, my_result):
    new_result = torch.zeros(table_size, table_size)
    for r, value in my_result:
        position = r.added_piece
        new_result[position[0], position[1]] = value
    tensor_sum = torch.sum(new_result)
    if abs(tensor_sum - 1) > 0.0001:
        print(f"Non ha normalization bene MCTS! {tensor_sum}")
        if tensor_sum == 0:
            tensor_sum = 1
        new_result = new_result / tensor_sum
    return new_result


def get_child_selected_by_nn(row, col, state):
    if not state.explored:
        state.explore()
    for child in state.children:
        if child.added_piece == (row, col):
            return child


def find_obvious_moves(tensor_state_value, first_player):
    def check_if_near_to_connected(rr, cc, tensor_index_to_check):
        near_a_minus_1 = False
        near_a_plus_1 = False
        if rr - 1 >= 0:
            if tensor_state_value[tensor_index_to_check, rr - 1, cc] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr - 1, cc] == 1:
                near_a_plus_1 = True
        if rr + 1 < board_size:
            if tensor_state_value[tensor_index_to_check, rr + 1, cc] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr + 1, cc] == 1:
                near_a_plus_1 = True
        if cc - 1 >= 0:
            if tensor_state_value[tensor_index_to_check, rr, cc - 1] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr, cc - 1] == 1:
                near_a_plus_1 = True
        if cc + 1 < board_size:
            if tensor_state_value[tensor_index_to_check, rr, cc + 1] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr, cc + 1] == 1:
                near_a_plus_1 = True
        if rr - 1 >= 0 and cc + 1 < board_size:
            if tensor_state_value[tensor_index_to_check, rr - 1, cc + 1] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr - 1, cc + 1] == 1:
                near_a_plus_1 = True
        if rr + 1 < board_size and cc - 1 >= 0:
            if tensor_state_value[tensor_index_to_check, rr + 1, cc - 1] == -1:
                near_a_minus_1 = True
            elif tensor_state_value[tensor_index_to_check, rr + 1, cc - 1] == 1:
                near_a_plus_1 = True
        if (tensor_index_to_check == 0 and rr == board_size - 1) or\
           (tensor_index_to_check == 1 and cc == board_size - 1):
            near_a_minus_1 = True
        if (tensor_index_to_check == 0 and rr == 0) or\
           (tensor_index_to_check == 1 and cc == 0):
            near_a_plus_1 = True
        return near_a_minus_1, near_a_plus_1

    def how_many_minus_1_around(rr, cc, tensor_index_to_check):
        counter_near_a_minus_1 = 0
        if rr - 1 >= 0 and tensor_state_value[2, rr - 1, cc] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr - 1, cc, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        if rr + 1 < board_size and tensor_state_value[2, rr + 1, cc] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr + 1, cc, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        if cc - 1 >= 0 and tensor_state_value[2, rr, cc - 1] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr, cc - 1, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        if cc + 1 < board_size and tensor_state_value[2, rr, cc + 1] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr, cc + 1, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        if rr - 1 >= 0 and cc + 1 < board_size and tensor_state_value[2, rr - 1, cc + 1] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr - 1, cc + 1, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        if rr + 1 < board_size and cc - 1 >= 0 and tensor_state_value[2, rr + 1, cc - 1] == 0:
            near_a_minus_1, _ = check_if_near_to_connected(rr + 1, cc - 1, tensor_index_to_check)
            if near_a_minus_1:
                counter_near_a_minus_1 += 1
        return counter_near_a_minus_1

    def how_many_plus_1_around(rr, cc, tensor_index_to_check):
        counter_near_a_plus_1 = 0
        if rr - 1 >= 0 and tensor_state_value[2, rr - 1, cc] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr - 1, cc, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        if rr + 1 < board_size and tensor_state_value[2, rr + 1, cc] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr + 1, cc, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        if cc - 1 >= 0 and tensor_state_value[2, rr, cc - 1] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr, cc - 1, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        if cc + 1 < board_size and tensor_state_value[2, rr, cc + 1] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr, cc + 1, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        if rr - 1 >= 0 and cc + 1 < board_size and tensor_state_value[2, rr - 1, cc + 1] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr - 1, cc + 1, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        if rr + 1 < board_size and cc - 1 >= 0 and tensor_state_value[2, rr + 1, cc - 1] == 0:
            _, near_a_plus_1 = check_if_near_to_connected(rr + 1, cc - 1, tensor_index_to_check)
            if near_a_plus_1:
                counter_near_a_plus_1 += 1
        return counter_near_a_plus_1

    board_size = tensor_state_value.shape[-1]
    empty_board = True
    other_will_win_r = None
    other_will_win_c = None
    """
    for r in range(board_size):
        for c in range(board_size):
            print(int(tensor_state_value[1, r, c]), end=" ")
        print()
    print("$"*120)
    """
    list_near_a_minus_1_me = []
    list_near_a_plus_1_me = []
    list_near_a_minus_1_other = []
    list_near_a_plus_1_other = []
    for r in range(board_size):
        for c in range(board_size):
            if tensor_state_value[2, r, c] == 0:
                # Not a piece there!
                if first_player:
                    near_a_minus_1_me, near_a_plus_1_me = check_if_near_to_connected(r, c, 0)
                    near_a_minus_1_other, near_a_plus_1_other = check_if_near_to_connected(r, c, 1)
                else:
                    near_a_minus_1_me, near_a_plus_1_me = check_if_near_to_connected(r, c, 1)
                    near_a_minus_1_other, near_a_plus_1_other = check_if_near_to_connected(r, c, 0)
                """
                if near_a_minus_1_me or near_a_plus_1_me or near_a_minus_1_other or near_a_plus_1_other:
                    print(f"near_a_minus_1_me: {near_a_minus_1_me}, near_a_plus_1_me: {near_a_plus_1_me}")
                    print(f"near_a_minus_1_other: {near_a_minus_1_other}, near_a_plus_1_other: {near_a_plus_1_other}")
                    print(f"{r} {c}")
                """
                if near_a_minus_1_me and near_a_plus_1_me:
                    # print("Obv you will win move!")
                    return r, c
                elif near_a_minus_1_me:
                    list_near_a_minus_1_me.append((r, c))
                elif near_a_plus_1_me:
                    list_near_a_plus_1_me.append((r, c))
                if near_a_minus_1_other and near_a_plus_1_other:
                    other_will_win_r = r
                    other_will_win_c = c
                elif near_a_minus_1_other:
                    list_near_a_minus_1_other.append((r, c))
                elif near_a_plus_1_other:
                    list_near_a_plus_1_other.append((r, c))
            else:
                empty_board = False
    if empty_board:
        # print("Obv starting move!")
        return board_size // 2, board_size // 2
    if other_will_win_r is not None:
        return other_will_win_r, other_will_win_c
    for el in list_near_a_minus_1_me:
        r = el[0]
        c = el[1]
        if first_player:
            if how_many_plus_1_around(r, c, 0) >= 2:
                #print(f"Obv me winning in 2 moves! [{r}; {c}]")
                return r, c
        else:
            if how_many_plus_1_around(r, c, 1) >= 2:
                #print(f"Obv me winning in 2 moves! [{r}; {c}]")
                return r, c
    for el in list_near_a_plus_1_me:
        r = el[0]
        c = el[1]
        if first_player:
            if how_many_minus_1_around(r, c, 0) >= 2:
                #print(f"Obv me winning in 2 moves! [{r}; {c}]")
                return r, c
        else:
            if how_many_minus_1_around(r, c, 1) >= 2:
                #print(f"Obv me winning in 2 moves! [{r}; {c}]")
                return r, c
    for el in list_near_a_minus_1_other:
        r = el[0]
        c = el[1]
        if first_player:
            if how_many_plus_1_around(r, c, 1) >= 2:
                #print(f"Obv other winning in 2 moves! [{r}; {c}]")
                return r, c
        else:
            if how_many_plus_1_around(r, c, 0) >= 2:
                #print(f"Obv other winning in 2 moves! [{r}; {c}]")
                return r, c
    for el in list_near_a_plus_1_other:
        r = el[0]
        c = el[1]
        if first_player:
            if how_many_minus_1_around(r, c, 1) >= 2:
                #print(f"Obv other winning in 2 moves! [{r}; {c}]")
                return r, c
        else:
            if how_many_minus_1_around(r, c, 0) >= 2:
                #print(f"Obv other winning in 2 moves! [{r}; {c}]")
                return r, c
    return None, None
