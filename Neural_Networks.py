import time
import torch
from torch import nn
import random
import matplotlib.pyplot as plt


class HexLinearModel(nn.Module):

    def __init__(self, board_size):

        super().__init__()
        self.board_size = board_size
        self.next_move_finder = nn.Sequential(
            # + 1 because we add the player playing
            nn.Linear(self.board_size*self.board_size + 1, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.board_size * self.board_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)
        # print(x[0])
        x = self.next_move_finder(x)
        # print(x.shape)
        # print(x[0])
        # print(torch.sum(x[0]))
        return x


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be an object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 model: torch.nn.Module):

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.model = to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastic Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(),
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

        x_batch = to_cuda(x_batch)
        y_batch = to_cuda(y_batch)

        # Perform the forward pass
        predictions = self.model(x_batch)

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

        return loss.detach().cpu().item()

    def train(self, epochs, data):
        """
        Trains the model for [self.epochs] epochs.
        """

        for epoch in range(epochs):
            # Perform a full pass through all the training samples
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

    def predict(self, tensor_state_value):
        prediction = self.model.forward(tensor_state_value)

        sum_prediction = torch.sum(prediction[tensor_state_value[:, 1:] == 0])
        if sum_prediction == 0:
            # The sum of the probability of the not occupied cell is zero, so we select a random free cell
            # print("The sum of probability on free cell was 0 but I'm smart so I select a random move:")
            # print(prediction)
            possibility = [i - 1 for i in range(1, tensor_state_value.shape[1]) if tensor_state_value[:, i] == 0]
            if len(possibility) == 0:
                print("Asked to predict a move but no free cell available!")
                print("tensor_state_value")
                print(tensor_state_value)
                print("prediction")
                print(prediction)
                print("tensor_state_value[:, 1:] != 0")
                print(tensor_state_value[:, 1:] != 0)
                exit()
            else:
                rand_index = random.randrange(len(possibility))
                prediction[tensor_state_value[:, 1:] != 0] = 0
                prediction[:, possibility[rand_index]] = 1.0
        else:
            prediction[tensor_state_value[:, 1:] != 0] = 0
            prediction = prediction/sum_prediction
        prediction = torch.reshape(prediction, (self.model.board_size, self.model.board_size))
        return prediction

    def plot_loss(self):
        plt.plot(list(range(len(self.train_history))), self.train_history)
        plt.show()
