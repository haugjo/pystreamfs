import time
import psutil
import os

import numpy as np

# load PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class CancelOut(nn.Module):
    '''
    CancelOut layer
    '''
    def __init__(self, input_size, *kargs, **kwargs):
        '''
        :param input_size:
        :param kargs:
        :param kwargs:
        '''
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, x):
        '''
        :param x: input
        :return: x * sigmoid(Weight)
        '''
        return x * torch.sigmoid(self.weights.float())

class myDataset(Dataset):
    '''

    Dataset Class for PyTorch model
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class NeuralNet(nn.Module):
    '''
    ANN
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        '''

        :param input_size: size of the input layer
        :param hidden_size: hidden size of ANN
        :param num_classes: output size of ANN
        '''

        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.CancelOut = CancelOut(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x1 = self.CancelOut(x)
        x2 = self.fc1(x1)
        x3 = self.relu(x2)
        x4 = self.fc2(x3)
        return x4


def train_ann(X, y, num_epochs):
    '''

    :param X:
    :param y:
    :param num_epochs:
    :return: CancelOut weights
    '''

    model = NeuralNet(X.shape[1], 10, 2)

    mydataset = myDataset(X, y)
    batch_size = 32
    learning_rate = 0.01
    train_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (sample, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sample.float())
            loss = criterion(outputs, labels.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return list(model.CancelOut.parameters())[0].detach().numpy()


# params for the run function
First_run = True
X_all, y_all = None, None

def run_nnfs(X, y, param):
    """Neural Network Feature Selection

    FS using Deep Learning

    :param numpy.nparray x: datapoint
    :param numpy.nparray y: class of the datapoint
    :param dict param: parameters

    :return: w (feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float


    TODO think about param
    """
    global First_run, X_all, y_all

    if First_run:
        First_run = False
        X_all = X
        y_all = y
    else:
        X_all = np.vstack((X_all, X))
        y_all = np.append(y_all, y)

    start_t = time.perf_counter()  # time taking

    w = train_ann(X_all, y_all, 10)

    return w, time.perf_counter() - start_t, psutil.Process(os.getpid()).memory_percent()
