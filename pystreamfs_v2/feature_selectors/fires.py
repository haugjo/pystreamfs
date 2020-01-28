from pystreamfs_v2.feature_selectors.base_feature_selector import BaseFeatureSelector
from pystreamfs_v2.utils.exceptions import InvalidModelError
from pystreamfs_v2.feature_selectors.fires_utils import monte_carlo_sampling, Net, SDT
import numpy as np
from scipy.stats import norm
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


class FIRESFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr, sigma_init=1, epochs=5, batch_size=20, lr_mu=0.1, lr_sigma=0.1,
                 model='probit', hidden_dim=100, hidden_layers=3, output_dim=2, mc_samples=5, lr_optimizer=0.01,
                 tree_depth=3, lamda=0.001):
        if model == 'probit':
            supports_multi_class = False
            supports_streaming_features = False
            supports_concept_drift_detection = True
        elif model == 'ann':
            supports_multi_class = True
            supports_streaming_features = False
            supports_concept_drift_detection = True
        elif model == 'sdt':
            supports_multi_class = True
            supports_streaming_features = False
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
        self.model = model

        # Neural Net model specific paramters
        if self.model in ['ann', 'sdt']:
            self.output_dim = output_dim
            self.mc_samples = mc_samples  # Monte Carlo samples
            self.lr_optimizer = lr_optimizer  # Learning Rate

            # ANN only
            self.hidden_layers = hidden_layers
            self.hidden_dim = hidden_dim

            # SDT only
            self.tree_depth = tree_depth
            self.num_inner_nodes = 2 ** tree_depth - 1
            self.lamda = lamda

            if self.model == 'ann':  # ANN
                self.mu_layer = torch.zeros((hidden_dim, n_total_ftr))  # mu of weights in first layer
                self.sigma_layer = torch.ones((hidden_dim, n_total_ftr))  # Todo: * sigma_init  # sigma of weigths in first layer
            else:  # SDT
                self.mu_layer = torch.zeros((self.num_inner_nodes, n_total_ftr))  # mu of weights in first layer
                self.sigma_layer = torch.ones((self.num_inner_nodes, n_total_ftr))  # Todo: * sigma_init  # sigma of weigths in first layer

    def weight_features(self, x, y):
        # Update estimates of mu and sigma given the model
        if self.model == 'probit':
            self.__probit(x, y)
        elif self.model in ['ann', 'sdt']:
            self.__nonlinear(x, y, self.model)
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

            # Transfer label 0 to label -1
            y[y == 0] = -1

            for i in range(0, len(y), self.batch_size):
                # Load mini batch
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Helper functions
                dot_mu_x = np.dot(x_batch, self.mu)
                rho = np.sqrt(1 + np.dot(x_batch**2, self.sigma**2))

                # Gradients
                nabla_mu = norm.pdf(y_batch/rho * dot_mu_x) * (y_batch/rho * x_batch.T)
                nabla_sigma = norm.pdf(y_batch/rho * dot_mu_x) * (- y_batch/(2 * rho**3) * 2 * (x_batch**2 * self.sigma).T * dot_mu_x)

                # Update parameters
                self.mu += self.lr_mu * np.mean(nabla_mu, axis=1)
                self.sigma += self.lr_sigma * np.mean(nabla_sigma, axis=1)

                # Limit sigma to range [0, inf]
                self.sigma[self.sigma < 0] = 0

    def __nonlinear(self, x, y, type):
        ########################################
        # 1. DATA PREPARATION
        ########################################
        # Format sample as tensor
        x = torch.from_numpy(x).float()  # format sample as tensor
        y = torch.from_numpy(y).long()

        ########################################
        # 2. MONTE CARLO SAMPLING
        ########################################
        if type == 'ann':  # ANN
            size = (self.hidden_dim, self.n_total_ftr)
            xavier = True
        else:  # SDT
            size = (self.num_inner_nodes, self.n_total_ftr)
            xavier = True  # Todo: Do we really want this?

        theta, epsilon = monte_carlo_sampling(mc_samples=self.mc_samples,  # Sample the first layer weights
                                              mu=self.mu,
                                              sigma=self.sigma,
                                              size=size,
                                              input_dim=self.n_total_ftr,
                                              output_dim=self.output_dim,
                                              xavier=xavier)

        ########################################
        # 3. TRAINING ANN OR SDT
        ########################################
        nabla_theta = dict()

        for l in range(self.mc_samples):  # For all samples L
            # Initialize gradient of theta for current sample l
            nabla_theta[l] = torch.zeros_like(theta[l])

            if type == 'ann':  # ANN
                # Initialize Neural Net, loss function and optimizer
                model = Net(self.n_total_ftr, self.hidden_dim, self.hidden_layers, self.output_dim)
                criterion = nn.NLLLoss()  # Negative Log Likelihood loss for classification
            else:  # SDT
                model = SDT(depth=self.tree_depth, lamda=self.lamda, input_dim=self.n_total_ftr, output_dim=self.output_dim)
                criterion = nn.CrossEntropyLoss()

            model.init_weights(theta[l].clone())  # set weights of current MC sample
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_optimizer)

            # ITERATIVE UPDATE
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
                    if type == 'ann':
                        y_pred = model(x_batch)
                        loss = criterion(y_pred, y_batch)
                    else:
                        y_pred, penalty = model(x_batch)
                        loss = criterion(y_pred, y_batch)
                        loss += penalty

                    # Perform backpropagation and update weights
                    loss.backward()
                    optimizer.step()

                    # Add gradient of current mini batch
                    if type == 'ann':  # ANN
                        nabla_theta[l] += model.linear_in.weight.grad
                    else:  # SDT
                        nabla_theta[l] += model.inner_nodes.linear.weight.grad

        ########################################
        # 4. COMPUTE GRADIENT ON MU AND SIGMA
        ########################################
        # Initialize the gradients
        nabla_mu = torch.zeros(theta[0].size())
        nabla_sigma = torch.zeros(theta[0].size())

        for l in range(self.mc_samples):
            # According to gradients in paper:
            nabla_mu += nabla_theta[l]
            nabla_sigma += nabla_theta[l] * epsilon[l]

        nabla_mu /= self.mc_samples  # average for L samples
        nabla_sigma /= self.mc_samples

        ########################################
        # 5. UPDATE MU AND SIGMA
        ########################################
        self.mu_layer -= self.lr_mu * nabla_mu
        self.sigma_layer -= self.lr_sigma * nabla_sigma

        # limit sigma to range [0, inf]
        self.sigma_layer[self.sigma_layer < 0] = 0

        ########################################
        # 6. AGGREGATE PARAMETERS
        ########################################
        if type == 'ann':  # ANN
            scaler = self.hidden_dim
        else:  # SDT
            scaler = self.num_inner_nodes

        self.mu = torch.sum(self.mu_layer, 0).numpy() / scaler
        self.sigma = torch.sum(self.sigma_layer, 0).numpy() / scaler

    def __update_weights(self):
        mu = self.mu.copy()
        sigma = self.sigma.copy()

        # Closed form solution of weight objective function
        self.raw_weight_vector = 0.5 * mu**2 - 0.5 * (np.dot(mu ** 2, sigma ** 2) / np.dot(sigma ** 2, sigma ** 2)) * sigma**2

        # Rescale to [0,1] -> we need positive weights for feature selection but want to maintain the rankings
        self.raw_weight_vector = MinMaxScaler().fit_transform(self.raw_weight_vector.reshape(-1, 1)).flatten()
