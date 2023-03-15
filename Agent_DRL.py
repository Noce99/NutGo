import copy
import os
import random
from datetime import datetime
import torch
from Neural_Networks import HexLinearModel, Trainer, to_cuda
from Hex import HexNode
from monte_carlo_tree_search import MCTS
from Agent import Agent


class HexAgent(Agent):

    def __init__(self, board_size, batch_size, learning_rate, max_num_of_batches):
        self.board_size = board_size
        self.dataset = []

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_num_of_batches = max_num_of_batches
        self.model = HexLinearModel(board_size)
        self.trainer = Trainer(self.batch_size, self.learning_rate, self.model)

    def train_while_playing(self, epochs, time_limit, simulations_limit, num_of_games, prob_of_random_move):
        red_was_the_winner = 0
        blue_was_the_winner = 0
        for nog in range(num_of_games):
            start_state = HexNode(True, [[None for _ in range(self.board_size)] for _ in range(self.board_size)],
                                  self.trainer, prob_of_random_move)
            state = start_state
            while True:
                new_state, result = self.get_a_move_from_mcts(HexNode(True, copy.deepcopy(state.state_value),
                                                              state.added_piece, self.get_a_move_from_nn,
                                                              prob_of_random_move), time_limit,
                                                              max_num_of_simulations=simulations_limit)
                self.add_data_to_dataset(copy.deepcopy(state.state_value), copy.deepcopy(state.first_player),
                                         copy.deepcopy(result))
                state = new_state
                if state.final_state:
                    if state.winner_first_player:
                        print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
                new_state, result = self.get_a_move_from_mcts(HexNode(False, copy.deepcopy(state.state_value),
                                                                      state.added_piece, self.get_a_move_from_nn,
                                                                      prob_of_random_move), time_limit,
                                                              max_num_of_simulations=simulations_limit)
                self.add_data_to_dataset(copy.deepcopy(state.state_value), copy.deepcopy(state.first_player),
                                         copy.deepcopy(result))
                state = new_state
                if state.final_state:
                    if state.winner_first_player:
                        print("Red Won!", end=" ")
                        red_was_the_winner += 1
                    else:
                        print("Blue Won!", end=" ")
                        blue_was_the_winner += 1
                    break
            self.trainer.train(epochs, self.dataset)
            print(
                f"{nog + 1}/{num_of_games} game finished {len(self.dataset) - 1} batches with {self.batch_size} data +"
                f" {int(self.dataset[-1][0].shape[0])}")
        print(f"Red/Blue wins -> {red_was_the_winner}/{blue_was_the_winner}")

    def add_data_to_dataset(self, state_value, first_player, result):
        data_x = from_state_value_to_1d_tensor(state_value, first_player)
        data_y = from_result_to_1d_tensor(len(state_value), result, first_player)
        if torch.sum(data_y) > 0.1:
            if len(self.dataset) != 0 and self.dataset[-1][0].shape[0] < self.batch_size:
                # There is some space for adding data in the last batch
                self.dataset[-1] = (torch.cat((self.dataset[-1][0], data_x), 0),
                                    torch.cat((self.dataset[-1][1], data_y), 0),)
            else:
                # There is NOT space for adding data in the last batch we create a new batch
                self.dataset.append((data_x, data_y))
                if len(self.dataset) > self.max_num_of_batches:
                    del self.dataset[0]

    def save_dataset(self):
        if len(self.dataset) == 0:
            print("Impossible to save an empty dataset!")
            exit()
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets")
        current_time = datetime.now()
        data_x = self.dataset[0][0]
        data_y = self.dataset[0][1]
        for d_x, d_y in self.dataset[1:]:
            data_x = torch.cat((data_x, d_x), 0)
            data_y = torch.cat((data_y, d_y), 0)
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
            num_of_batches = num_of_data // self.batch_size + 1
            self.dataset = []
            for b in range(num_of_batches):
                self.dataset.append((data_x[b*self.batch_size:(b+1)*self.batch_size, :],
                                     data_y[b*self.batch_size:(b+1)*self.batch_size, :]))
            self.print_how_much_data()
            return None
        elif not os.path.exists("./datasets/" + dataset_name + ".X"):
            print(f"[{'./datasets/' + dataset_name + '.X'}] doesn't exists!")
        elif not os.path.exists("./datasets/" + dataset_name + ".Y"):
            print(f"[{'./datasets/' + dataset_name + '.Y'}] doesn't exists!")
        exit()

    def save_weight(self):
        if not os.path.exists("./weights"):
            os.mkdir("./weights")
        current_time = datetime.now()
        torch.save(self.model.state_dict(), f"./weights/{self.board_size}x{self.board_size}_"
                                            f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.w")

    def load_weight(self, weight_name):
        if os.path.exists("./weights/"+weight_name):
            self.model.load_state_dict(torch.load("./weights/"+weight_name))
            # self.model.eval()
        else:
            print(f"[{'./weights/'+weight_name}] doesn't exists!")
            exit()

    def get_a_move_from_nn(self, state_value, first_player):
        tensor_state_value = to_cuda(from_state_value_to_1d_tensor(state_value, first_player))
        prediction = self.trainer.predict(tensor_state_value)
        random_float = random.random()
        actual_sum = 0
        for r in range(prediction.shape[0]):
            for c in range(prediction.shape[0]):
                actual_sum += prediction[r, c]
                if random_float <= actual_sum:
                    return r, c, torch.flatten(prediction)
        print(f"Actual Sum: {actual_sum}")
        print(f"Random Float: {random_float}")
        print("Prediction:")
        print(prediction)
        print("State Value:")
        print(state_value)
        print("Tensor state value:")
        print(tensor_state_value)
        assert "Impossible to arrive there!"

    @staticmethod
    def get_a_move_from_mcts(actual_s, time, max_num_of_simulations=-1):
        my_mcts = MCTS(actual_s, time, max_num_of_simulations)
        result = my_mcts.explore(verbose=False)
        best_child = result[0][0]
        max_value = result[0][1]
        for i, r in enumerate(result[1:]):
            if r[1] > max_value:
                best_child = r[0]
                max_value = r[1]
        return best_child, result

    def get_move(self, _, state_value, first_player):
        return self.get_a_move_from_nn(state_value, first_player)

    def print_how_much_data(self):
        print(f"I have {len(self.dataset)} batches!")
        for i, batch in enumerate(self.dataset):
            print(f"{i+1}: [{batch[0].shape}, {batch[1].shape}]")

    def plot_accuracy(self):
        correct = 0
        n = 0
        for x_batch, y_batch in self.dataset:
            for i in range(x_batch.shape[0]):
                y = torch.argmax(y_batch[i])
                predicted_tensor = torch.flatten(self.trainer.predict(to_cuda(x_batch[i][None, :])))
                predicted_y = torch.argmax(predicted_tensor)
                """
                print("Y:")
                print(y_batch[i])
                print(f"the maximum is in {y}")
                print("Predicted_Y:")
                print(predicted_tensor)
                print(f"the maximum is in {predicted_y}")
                print(torch.sum(y_batch[i]))
                print(torch.sum(predicted_tensor))
                exit()
                """
                if y == predicted_y:
                    correct += 1
                n += 1
        print(f"Accuracy: {correct/n}")


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


def from_result_to_1d_tensor(table_size, my_result, first_player):
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
