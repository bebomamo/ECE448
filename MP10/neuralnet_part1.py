# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part1. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256
        # TODO Define the network architecture (layers) based on these specifications.
        # very similar code to Amnon
        self.l1 = nn.Linear(in_features=2883, out_features=100, bias=True) #linear layer 1
        self.l2 = nn.Linear(in_features=100, out_features=50, bias=True) #linear layer 2
        self.out = nn.Linear(in_features=50, out_features=4, bias=False) #output layer
        self.lrate = lrate #from tutorial sources
        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate, momentum=0.9) #want to use an optimizer instead of doing this explicitely
        self.criterion = nn.CrossEntropyLoss() #storing the loss function from torch.nn instead of explicitely writing it

        # raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        # again very similar code to Amnon and the tutorials
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)
        return x
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
        self.optimizer.zero_grad() #clear the SGD optimizer from last batch
        #forward propagate --> backward propagate --> optimize/update weights
        output = self(x) #gets outputs(calls forward function)
        loss = self.criterion(output, y) #gets loss
        loss.backward() #back prop with given loss calculating gradients
        self.optimizer.step() #update weights with SGD

        return loss.item()

        # raise NotImplementedError("You need to write this part!")
        # # Important, detach and move to cpu before converting to numpy and then to python float.
        # # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        # return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    net = NeuralNet(0.00007, nn.CrossEntropyLoss(), len(train_labels), len(dev_set))
    batch_loader = torch.utils.data.DataLoader(get_dataset_from_arrays(train_set, train_labels), batch_size=batch_size, shuffle=False, drop_last=True)
    # Supposed to do data standardization but I dont really know how yet
    # ^ STILL DIDN'T DO THIS BUT INSTEAD LOWERED LEARNING RATE TO ACCOUNT FOR THE NON-NORMALIZED PIXEL DATA
    #training epochs
    losses = []
    for epoch in range(epochs):
        print("\rEpoch {}".format(epoch), end="")
        epoch_loss = 0.0
        for batch in batch_loader:
            batch_features = batch["features"]
            batch_labels = batch["labels"]
            epoch_loss += net.step(batch_features, batch_labels)
        epoch_loss /= len(batch_loader)
        print("\rEpoch {}, Running loss {}".format(epoch, epoch_loss), end="")
        losses.append(epoch_loss)

    outputs = net(dev_set)
    _, predicted = torch.max(outputs, 1) #got this line from pytorch tutorial
    predicted = predicted.cpu().detach().numpy()

    return losses, predicted, net
    # raise NotImplementedError("You need to write this part!")
    # # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    # return [],[],None
