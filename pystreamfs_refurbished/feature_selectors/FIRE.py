from pystreamfs_refurbished.feature_selectors import BaseFeatureSelector
import numpy as np
from scipy.stats import norm


class FIREFeatureSelector(BaseFeatureSelector):
    def __init__(self, total_ftr, sigma_init, epochs, batch_size):
        super().__init__(self, total_ftr, True, True)

        # Initialize theta ~ N(mu, sigma)
        self.mu = np.zeros(total_ftr)
        self.sigma = np.ones(total_ftr) * sigma_init
        self.epochs = epochs
        self.batch_size = batch_size


    def select_features(self, x, y):
        mu = self.mu.copy()  # Todo: is this step necessary?
        sigma = self.sigma.copy()

        for epoch in range(self.epochs):  # SGD
            np.random.shuffle(x)

            for i in range(0, x.shape[0], self.batch_size):
                # Load mini batch
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Sub-functions
                dot_x_mu = np.dot(x, mu)  # x . mu
                dot_x_sigma = np.dot(x ** 2, sigma ** 2)  # x^2 . sigma^2

                # Partial derivatives of log likelihood with respect to mu and sigma
                nabla_mu = norm.pdf(dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * (
                        x.T / np.sqrt(1 + dot_x_sigma))

                nabla_sigma = norm.pdf((-1) ** (1 - y) * dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * (
                        (2 * x ** 2 * sigma).T * dot_x_mu) / (-2 * np.sqrt(1 + dot_x_sigma) ** 3)

                # update parameters
                mu += param['lr_mu'] * np.mean(nabla_mu, axis=1)
                sigma += param['lr_sigma'] * np.mean(nabla_sigma, axis=1)

        # update param
        param['mu'] = mu
        param['sigma'] = sigma

        # Update weights
        w_unscaled, param = _update_weights(mu.copy(), sigma.copy(), param, X.shape[1])

        # scale weights to [0,1] because pystreamfs considers absolute weights for feature selection
        w = MinMaxScaler().fit_transform(w_unscaled.reshape(-1, 1)).flatten()

        # concept drift detection
        if param['check_drift'] is True:
            param = _check_concept_drift(mu, sigma, param)

        return w, param

    def detect_concept_drift(self, x, y):
        raise NotImplementedError
