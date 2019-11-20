from pystreamfs_v2.feature_selectors.base_feature_selector import BaseFeatureSelector
from pystreamfs_v2.utils.exceptions import InvalidModelError
import numpy as np
from scipy.stats import norm
import torch
from torch import nn


class FIRESFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr, sigma_init=1, epochs=5, batch_size=20, lr_mu=0.1, lr_sigma=0.1,
                 lr_weights=0.1, lr_lamb=0.1, lamb_init=1, model='probit', hidden_dim=100, output_dim=2, mc_samples=10, lr_optimizer=0.01):
        if model == 'probit':
            supports_multi_class = False
            supports_streaming_features = False
            supports_concept_drift_detection = True
        elif model == 'neural_net':
            supports_multi_class = True
            supports_streaming_features = True
            supports_concept_drift_detection = True
        else:
            print('FIRES model does not exist!')
            raise NotImplementedError  # Todo: look for different error

        super().__init__('FIRES', n_total_ftr, n_selected_ftr, supports_multi_class, supports_streaming_features, supports_concept_drift_detection)

        self.mu = np.zeros(n_total_ftr)
        self.sigma = np.ones(n_total_ftr) * sigma_init
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.lr_weights = lr_weights
        self.lr_lamb = lr_lamb
        self.lamb = lamb_init
        self.model = model

        # Neural Net model specific paramters
        if self.model == 'neural_net':
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.input_dim = n_total_ftr
            self.mc_samples = mc_samples
            self.lr_optimizer = lr_optimizer
            self.mu_layer = torch.zeros((hidden_dim, n_total_ftr))  # mu of weigths in first layer
            self.sigma_layer = torch.ones((hidden_dim, n_total_ftr))  # Todo: * sigma_init  # sigma of weigths in first layer

    def weight_features(self, x, y):
        # Update estimates of mu and sigma given the model
        if self.model == 'probit':
            self.__probit(x, y)
        elif self.model == 'neural_net':
            self.__neural_net(x, y)
        else:
            raise InvalidModelError('FIRE Feature Selection: The chosen model is not specified.')

        # Update feature weights
        self.__update_weights()

    def detect_concept_drift(self, x, y):
        raise NotImplementedError

    def __probit(self, x, y):
        for epoch in range(self.epochs):  # SGD
            # Shuffle sample
            random_idx = np.random.permutation(len(y))
            x = x[random_idx]
            y = y[random_idx]

            for i in range(0, len(y), self.batch_size):
                # Load mini batch
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Sub-functions
                dot_x_mu = np.dot(x_batch, self.mu)  # x . mu
                dot_x_sigma = np.dot(x_batch ** 2, self.sigma ** 2)  # x^2 . sigma^2

                # Partial derivatives of log likelihood with respect to mu and sigma
                nabla_mu = norm.pdf(dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y_batch) * (
                        x_batch.T / np.sqrt(1 + dot_x_sigma))

                nabla_sigma = norm.pdf((-1) ** (1 - y_batch) * dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y_batch) * (
                        (2 * x_batch ** 2 * self.sigma).T * dot_x_mu) / (-2 * np.sqrt(1 + dot_x_sigma) ** 3)

                # update parameters
                self.mu += self.lr_mu * np.mean(nabla_mu, axis=1)
                self.sigma += self.lr_sigma * np.mean(nabla_sigma, axis=1)

    def __neural_net(self, x, y):
        ########################################
        # 1. DATA PREPARATION
        ########################################
        # Dimensionality of Y
        output_dim = len(np.unique(y))  # Todo check if this can be retrieved at different point

        # Format sample as tensor
        x = torch.from_numpy(x).float()  # format sample as tensor
        y = torch.from_numpy(y).long()

        ########################################
        # 2. CHANGING FEATURES AND/OR CLASSES
        ########################################
        # Detect new features
        new_features = 0  # number of new features
        if x.size()[1] > self.input_dim:
            # mu_1, sigma_1, param, new_features = _new_input_dim(x.size()[1], mu_1, sigma_1, param)  # init new input nodes Todo
            print('New feature detected')

        # Detect new classes
        if len(y.unique()) > self.output_dim:
            # mu_2, sigma_2, param = _new_output_dim(output_dim, mu_2, sigma_2, param)  # init new output nodes Todo
            print("New class detected")

        ########################################
        # 3. MONTE CARLO SAMPLING
        ########################################
        theta, epsilon = self._monte_carlo_sampling((self.input_dim, self.hidden_dim))  # Sample first layer weights

        ########################################
        # 4. TRAINING THE NET
        ########################################
        nabla_theta = dict()

        for l in range(self.mc_samples):  # For all samples L
            # Initialize gradient of theta for current sample l
            nabla_theta[l] = torch.zeros(theta[l].size())

            # Initialize Neural Net, loss function and optimizer
            model = _Net(self.input_dim, self.hidden_dim, self.output_dim)  # Todo allow multiple hidden layers
            model.init_weights(theta[l].clone())  # set weights theta
            criterion = nn.NLLLoss()  # Negative Log Likelihood loss for classification
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_optimizer)  # specify optimizer, here SGD

            for epoch in range(self.epochs):
                # shuffle sample
                idx = torch.randperm(len(y))
                x = x[idx]
                y = y[idx]

                for i in range(0, len(y), self.batch_size):
                    # Load mini batch
                    x_batch = x[i:i + self.batch_size]
                    y_batch = y[i:i + self.batch_size]

                    # Zero Gradients
                    optimizer.zero_grad()

                    # Forward pass of neural net
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)

                    # Perform backpropagation and update weights
                    loss.backward()
                    optimizer.step()

                    if model.linear1.weight.grad.sum().item() == 0:  # TODO: delete when certain
                        print('Gradient is zero at')

                    # Add gradient of current mini batch
                    nabla_theta[l] += model.linear1.weight.grad

            # average gradients for epochs and mini-batches
            nabla_theta[l] /= (self.epochs * self.batch_size)

        ########################################
        # 5. COMPUTE GRADIENT ON MU AND SIGMA
        ########################################
        # Initialize the gradients
        nabla_mu = torch.zeros(theta[0].size())
        nabla_sigma = torch.zeros(theta[0].size())

        for l in range(self.mc_samples):
            # According to gradients in paper:
            nabla_mu += nabla_theta[l]
            nabla_sigma += nabla_theta[l] * epsilon[l]  # Todo: check if multiplication works

        nabla_mu /= self.mc_samples  # average for L samples
        nabla_sigma /= self.mc_samples

        ########################################
        # 6. UPDATE MU AND SIGMA
        ########################################
        self.mu_layer -= self.lr_mu * nabla_mu
        self.sigma_layer -= self.lr_sigma * nabla_sigma

        # minimal value of sigma is 0
        # self.sigma_layer[self.sigma_layer < 0] = 0  # Todo: check if necessary

        # Merge mu and sigma matrix to 1-D vectors TODO: CONTINUE WITH CHECK OF NN
        self.mu = torch.mean(self.mu_layer, 0)  # check if correct mean is build
        self.sigma = torch.mean(self.sigma_layer, 0)

    def __update_weights(self):
        mu = self.mu.copy()
        sigma = self.sigma.copy()

        # Scale mu and sigma -> to avoid exploding gradient
        mu /= np.sum(np.abs(mu))
        sigma /= np.sum(np.abs(sigma))

        # Compute derivative of weight and lambda +  update with gradient ascent
        w_update = np.abs(mu) - 2 * self.lamb * (self.raw_weight_vector * sigma ** 2) - 2 * self.raw_weight_vector
        self.raw_weight_vector += self.lr_weights * w_update

        lamb_update = -np.dot(self.raw_weight_vector ** 2, sigma ** 2)
        self.lamb += self.lr_lamb * lamb_update

    ########################################
    # Helper Functions for Neural Net Todo: consider move to separate file
    ########################################
    def _monte_carlo_sampling(self, size):
        """
        Monte Carlo sampling of theta with reparameterization trick

        size = (rows, columns) of weight matrix theta
        """
        theta = dict()
        epsilon = dict()  # reparametrization parameter

        for l in range(self.mc_samples):
            # Xavier weight initialization
            epsilon[l] = torch.distributions.normal.Normal(0, np.sqrt(2 / (self.input_dim + self.output_dim))).sample((size[0], size[1]))
            theta[l] = self.sigma * epsilon[l] + self.mu

        return theta, epsilon

    def _new_input_dim(self, new_dim, mu_1, sigma_1, param):
        """
        Adjust to new dimensionality of feature space (Feature Stream) by adding new input nodes.
        Initialize new first layer weights.

        :param int new_dim: new input dimensionality
        :param torch.tensor mu_1: mu of first layer weights
        :param torch.tensor sigma_1: sigma of first layer weights
        :param dict param: parameters, includes:
            - int d: prior input dimensionality
        :return: mu_1, sigma_1 (including the new input nodes), param, new_features (no. of added input nodes/new features)
        :rtype: torch.tensor, torch.tensor, dict, int
        """
        # number of new features
        new_features = new_dim - param['d']

        # current average of mu and sigma
        avg_mu = torch.mean(mu_1, 1).view(-1, 1)
        avg_sigma = torch.mean(sigma_1, 1).view(-1, 1)

        cat_mu = torch.cat([avg_mu] * new_features, 1)
        cat_sigma = torch.cat([avg_sigma] * new_features, 1)

        # add input node
        mu_1 = torch.cat((mu_1, cat_mu), 1)
        sigma_1 = torch.cat((sigma_1, cat_sigma), 1)

        param['d'] = new_dim  # update feature dimensionality

        return mu_1, sigma_1, param, new_features

    def _new_output_dim(new_dim, mu_2, sigma_2, param):
        """
        Adjust to newly appeared classes (Concept Evolution) by adding new output nodes.
        Initialize new second layer weights.

        :param int new_dim: new output dimensionality
        :param torch.tensor mu_2: mu of second layer weights
        :param torch.tensor sigma_2: sigma of second layer weights
        :param dict param: parameters, includes:
            - int classes: prior no. of classes
        :return: mu_2, sigma_2 (including the new output nodes, param
        :rtype: torch.tensor, torch.tensor, dict, int
        """
        # number of new classes
        new_classes = new_dim - param['classes']

        # current average of mu and sigma
        avg_mu = torch.mean(mu_2, 0)
        avg_sigma = torch.mean(sigma_2, 0)

        cat_mu = torch.unsqueeze(torch.cat([avg_mu] * new_classes, 0), 0)
        cat_sigma = torch.unsqueeze(torch.cat([avg_sigma] * new_classes, 0), 0)

        # add input node
        mu_2 = torch.cat((mu_2, cat_mu), 0)
        sigma_2 = torch.cat((sigma_2, cat_sigma), 0)

        param['classes'] = new_dim  # update number of classes

        return mu_2, sigma_2, param


class _Net(nn.Module):
    """Feed Forward Neural net with fully connected layers
    - with softplus activation of the hidden nodes
    - with softmax activation of the output nodes
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(_Net, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)  # define input to hidden layer
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)  # define hidden to output layer
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        h_linear = self.linear1(x)
        h_activation = self.softplus(h_linear)
        out_linear = self.linear2(h_activation)
        y_pred = self.logsoftmax(out_linear)
        return y_pred

    def init_weights(self, theta):
        # initialize weights of first layer
        self.linear1.weight = nn.Parameter(theta)
