import time
from collections import OrderedDict

import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from Utils import my_argmax


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class HexConvolutionalModel(nn.Module):

    def __init__(self, board_size, hidden_layers=0, neuron_per_layers=64, activation_function="RELU"):
        self.board_size = board_size
        super().__init__()
        activation_functions = {"linear": None, "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "RELU": nn.ReLU()}
        if self.board_size < 6:
            layers = [("0", nn.Linear(self.board_size * self.board_size + 1, neuron_per_layers))]
            for hl in range(hidden_layers):
                layers.append((f"activation_{hl}", activation_functions[activation_function]))
                layers.append((f"hl_{hl}", nn.Linear(neuron_per_layers, neuron_per_layers)))
            layers.append(("1", activation_functions[activation_function]))
            layers.append(("2", nn.Linear(neuron_per_layers, self.board_size * self.board_size)))
            layers.append(("3", nn.Softmax(dim=1)))
            self.next_move_finder = nn.Sequential(OrderedDict(layers))
            """
                nn.Linear(self.board_size*self.board_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.board_size * self.board_size),
                nn.Softmax(dim=1)
            )
            """
        else:
            self.next_move_finder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3), padding=0),
                nn.ReLU(),
            )
        for p_name, p in self.named_parameters():
            p.data = torch.randn(p.shape) * 0.2  # Random weight initialization
            p.requires_grad = True  # Not Freeze

    def forward(self, x):
        if self.board_size < 6:
            player = x[:, 0, 0, 0]
            x = x[:, 2, :, :]
            x = torch.concat([torch.transpose(player[None, :], 0, 1), torch.flatten(x, 1)], 1)
        out = self.next_move_finder(x)
        if self.board_size < 6:
            out = torch.reshape(out, (out.shape[0], self.board_size, self.board_size))
            out = out[:, None, :, :]
        return out


"""
def to_cuda(elements):
    # Transfers every object in elements to GPU VRAM if available.
    # elements can be an object or list/tuple of objects
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements
"""


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 model: torch.nn.Module,
                 optimizer="SGD"):

        optimizers = {"SGD": torch.optim.SGD,
                      "Adagrad": torch.optim.Adagrad,
                      "RMSProp": torch.optim.RMSprop,
                      "Adam": torch.optim.Adam}

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.model = self.model
        # print(self.model)

        # Define our optimizer. SGD = Stochastic Gradient Descent
        self.optimizer = optimizers[optimizer](self.model.parameters(),
                                               self.learning_rate)

        # Tracking loss
        self.train_history = []
        self.train_history_step = 0

    def train_step(self, x_batch, y_batch):
        """
        Perform forward, backward and gradient descent step here.
            :param x_batch:
            :param y_batch:
        """
        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        # TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
        """
        for i in range(y_batch.shape[0]):
            r_max, c_max = my_argmax(y_batch[i])
            y_batch[i] = torch.zeros_like(y_batch[i])
            y_batch[i, r_max, c_max] = 1.0
        """

        # Perform the forward pass
        predictions = self.model.forward(x_batch)
        predictions = predictions[:, 0, :, :]

        ###################################################################
        """
        print("True:")
        print(y_batch[0])
        print(torch.sum(y_batch[0]))
        print("Prediction")
        print(predictions[0])
        print(torch.sum(predictions[0]))
        """
        ###################################################################

        # Compute the cross entropy loss for the batch
        loss = self.loss_criterion(predictions, y_batch)
        # Backpropagation
        loss.backward()
        # Gradient descent step
        self.optimizer.step()

        return loss.detach().item()

    def train(self, epochs, data):
        """
        Trains the model for [self.epochs] epochs.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            epoch_steps = 0
            for x_batch, y_batch in data:
                loss = self.train_step(x_batch, y_batch)
                total_loss += loss
                epoch_steps += 1
            if epoch_steps == 0:
                print("Asked to training but no data in the database!")
                exit()
            average_loss = total_loss / epoch_steps
            self.train_history.append(average_loss)
            self.train_history_step += 1
        self.model.eval()

    def predict(self, tensor_state_value):
        prediction = self.model.forward(tensor_state_value[None, :])[0, 0]

        pieces = tensor_state_value[2]
        board_size = pieces.shape[0]

        sum_prediction = torch.sum(prediction[pieces == 0])
        # print(prediction)

        if sum_prediction == 0:
            # The sum of the probability of the not occupied cell is zero, so we select a random free cell
            # print("The sum of probability on free cell was 0 but I'm smart so I select a random move:")
            # print(prediction)
            possibility = []
            for row in range(board_size):
                for col in range(board_size):
                    if pieces[row, col] == 0:
                        possibility.append((row, col))
            if len(possibility) == 0:
                print("Asked to predict a move but no free cell available!")
                exit()
            else:
                rand_index = random.randrange(len(possibility))
                prediction = torch.zeros(board_size, board_size)
                prediction[possibility[rand_index][0], possibility[rand_index][1]] = 1.0
        else:
            prediction[pieces != 0] = 0
            prediction = prediction/sum_prediction
        return prediction

    def plot_loss(self):
        plt.plot(list(range(len(self.train_history))), self.train_history)
        plt.show()
