from pystreamfs.feature_selectors.base_feature_selector import BaseFeatureSelector
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class CancelOutFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr):
        super().__init__('CancelOut', n_total_ftr, n_selected_ftr, False, False, False)

    def weight_features(self, x, y):
        self.raw_weight_vector = self._train_ann(x, y, 50)

    @staticmethod
    def _train_ann(x, y, num_epochs):
        model = NeuralNet(x.shape[1], x.shape[1] + 1, 2)

        n_features = x.shape[1]
        mydataset = MyDataset(x, y)
        batch_size = int(n_features / 5)

        train_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam([
            {"params": model.CancelOut.parameters(), "lr": 0.01},
            {"params": model.fc1.parameters(), "lr": 0.003},
            {"params": model.fc2.parameters(), "lr": 0.003},
        ])
        patience = 3

        early_stopping = EarlyStopping(patience=patience, verbose=False)

        avg_train_losses = []
        train_losses = []

        for epoch in range(num_epochs):
            for i, (sample, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(sample.float())

                weights_co = list(model.CancelOut.parameters())[0]
                reg = torch.var(weights_co)
                nrm = torch.norm(weights_co, 1)  # torch.sum(torch.abs(weights_co))

                loss = criterion(outputs, labels.long()) - 0.001 * (reg / n_features) + 0.0001 * (nrm / n_features)
                train_losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # data for early stoping
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                # print("Early stopping")
                break

        return list(model.CancelOut.parameters())[0].detach().numpy()

    def detect_concept_drift(self, x, y):
        raise NotImplementedError


class NeuralNet(nn.Module):
    """ANN"""

    def __init__(self, input_size, hidden_size, num_classes):
        """
        :param input_size: size of the input layer
        :param hidden_size: hidden size of ANN
        :param num_classes: output size of ANN
        """

        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU6()
        self.CancelOut = CancelOut(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x1 = self.CancelOut(x)
        x2 = self.fc1(x1)
        x3 = self.relu(x2)
        x4 = self.fc2(x3)
        return x4


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # https: // github.com / Bjarten / early - stopping - pytorch

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            #if self.verbose:
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class CancelOut(nn.Module):
    """CancelOut layer"""

    def __init__(self, input_size, *kargs, **kwargs):
        """
        :param input_size:
        :param kargs:
        :param kwargs:
        """
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, x):
        """
        :param x: input
        :return: x * sigmoid(Weight)
        """
        return x * torch.sigmoid(self.weights.float()+2)


class MyDataset(Dataset):
    """
    Dataset Class for PyTorch model
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
