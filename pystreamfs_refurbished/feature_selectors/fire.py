from pystreamfs_refurbished.feature_selectors.base_feature_selection import BaseFeatureSelector
from pystreamfs_refurbished.exceptions import InvalidModelError
import numpy as np
from scipy.stats import norm


class FIREFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr, sigma_init, epochs, batch_size, lr_mu, lr_sigma, lr_weights, lr_lamb, lamb_init, model='probit'):
        super().__init__(n_total_ftr, n_selected_ftr, False, True, True)

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

    def weight_features(self, x, y, active_features):  # Todo: handle active features
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
            np.random.shuffle(x)

            for i in range(0, x.shape[0], self.batch_size):
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
        raise NotImplementedError

    def __update_weights(self):
        mu = self.mu.copy()
        sigma = self.sigma.copy()

        # Scale mu and sigma -> to avoid exploding gradient
        mu /= np.sum(np.abs(mu))
        sigma /= np.sum(np.abs(sigma))

        # Compute derivative of weight and lambda +  update with gradient ascent
        w_update = np.abs(mu) - 2 * self.lamb * (self.weights * sigma ** 2) - 2 * self.weights
        self.weights += self.lr_weights * w_update

        lamb_update = -np.dot(self.weights ** 2, sigma ** 2)
        self.lamb += self.lr_lamb * lamb_update
