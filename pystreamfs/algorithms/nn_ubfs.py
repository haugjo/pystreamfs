import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def run_nn_ubfs(X, Y, param, **kw):
    """
    Uncertainty Based Feature Selection with Neural Net

    :param numpy.ndarray X: current data batch
    :param numpy.ndarray Y: labels of current batch
    :param dict param: parameters, this includes...
        int epochs: number of epochs (iterations over X)
        int mini_batch_size: no. of samples per mini_batch
        float lr_mu: learning rate for the update of mean
        float lr_sigma: learning rate for the update of standard deviation
    :return: w (feature weights), param
    :rtype numpy.ndarray, dict

    ...Todo: add to README.md
    """

    # Format sample as tensor
    x = torch.from_numpy(X).float()  # format sample as tensor
    y = torch.from_numpy(Y).view(-1, 1).float()

    # initialize uncertainty parameters for distribution of feature weights theta
    if 'mu_1' not in param and 'sigma_1' not in param:
        param['mu_1'] = torch.zeros((param['h'], x.size()[1]))  # input to hidden layer weights
        param['mu_2'] = torch.zeros((y.size()[1], param['h']))  # hidden to output layer weights

        param['sigma_1'] = torch.ones((param['h'], x.size()[1]))  # input to hidden layer weights
        param['sigma_2'] = torch.ones((y.size()[1], param['h']))  # hidden to output layer weights

        param['d'] = x.size()[1]  # initial dimensionality of feature space

    mu_1 = param['mu_1'].clone()
    mu_2 = param['mu_2'].clone()
    sigma_1 = param['sigma_1'].clone()
    sigma_2 = param['sigma_2'].clone()

    # Detect new features
    new_feature = False

    if x.size()[1] > param['d']:
        # add input node
        mu_1 = torch.cat((mu_1, torch.zeros((param['h'], 1))), 1)
        sigma_1 = torch.cat((sigma_1, torch.ones((param['h'], 1))), 1)

        param['d'] = x.size()[1]  # update feature dimensionality
        new_feature = True

    # Sample theta with Monte Carlo
    theta_1, theta_2, r_1, r_2 = _monte_carlo_sampling(x.size()[1], y.size()[1], mu_1, mu_2, sigma_1, sigma_2, param)

    # Gradient of theta
    nabla_theta_1 = dict()
    nabla_theta_2 = dict()

    for l in range(param['L']):
        # Gradient of theta for l
        nabla_theta_1[l] = torch.zeros(theta_1[l].size())
        nabla_theta_2[l] = torch.zeros(theta_2[l].size())

        # Initialize Neural Net, loss function and optimizer
        model = _Net(x.size()[1], param['h'], y.size()[1])
        model.init_weights(theta_1[l].clone(), theta_2[l].clone())  # set weights with theta
        criterion = nn.BCELoss()  # Cross entropy loss for classification
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # specify optimizer, here SGD

        for epoch in range(param['epochs']):
            # shuffle sample
            idx = torch.randperm(x.size()[0])
            x = x[idx]
            y = y[idx]

            for i in range(0, x.size()[0], param['mini_batch_size']):
                # Load mini batch
                x_b = x[i:i+param['mini_batch_size']]
                y_b = y[i:i+param['mini_batch_size']]

                # Zero Gradients
                optimizer.zero_grad()

                # Forward pass of neural net
                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)

                # Perform backpropagation and update weights
                loss.backward()
                optimizer.step()

                if model.linear1.weight.grad.sum().item() == 0:  # TODO: delete when certain
                    print('Gradient is zero at t={}'.format(param['t']))

                # Add gradient of current mini batch
                nabla_theta_1[l] += model.linear1.weight.grad
                nabla_theta_2[l] += model.linear2.weight.grad

        # average gradients for epochs and mini-batches
        nabla_theta_1[l] /= (param['epochs'] * param['mini_batch_size'])
        nabla_theta_2[l] /= (param['epochs'] * param['mini_batch_size'])

    # Compute gradient on mu and sigma
    nabla_mu_1 = torch.zeros(nabla_theta_1[0].size())
    nabla_mu_2 = torch.zeros(nabla_theta_2[0].size())

    nabla_sigma_1 = torch.zeros(nabla_theta_1[0].size())
    nabla_sigma_2 = torch.zeros(nabla_theta_2[0].size())

    for l in range(param['L']):
        nabla_mu_1 += nabla_theta_1[l]
        nabla_mu_2 += nabla_theta_2[l]

        nabla_sigma_1 += nabla_theta_1[l] * r_1[l]
        nabla_sigma_2 += nabla_theta_2[l] * r_2[l]

    nabla_mu_1 /= param['L']  # average for L samples
    nabla_mu_2 /= param['L']
    nabla_sigma_1 /= param['L']
    nabla_sigma_2 /= param['L']

    # Update mu and sigma
    mu_1 -= param['lr_mu'] * nabla_mu_1
    mu_2 -= param['lr_mu'] * nabla_mu_2
    sigma_1 -= param['lr_sigma'] * nabla_sigma_1
    sigma_2 -= param['lr_sigma'] * nabla_sigma_2

    # Update param
    param['mu_1'] = mu_1
    param['mu_2'] = mu_2
    param['sigma_1'] = sigma_1
    param['sigma_2'] = sigma_2

    # Update weights
    mu_2_norm = torch.abs(mu_2)/torch.sum(torch.abs(mu_2))  # normalize weights of layer 2
    sigma_2_norm = torch.abs(sigma_2)/torch.sum(torch.abs(sigma_2))

    mu = mu_1 * mu_2_norm.t()  # average layer 1 weights relative to layer 2 weight
    mu = torch.sum(mu, dim=0).numpy()
    sigma = sigma_1 * sigma_2_norm.t()
    sigma = torch.sum(sigma, dim=0).numpy()

    # TOdo: delete when certain
    if (sigma < 0).any():
        print('Sigma is negative at t={}'.format(param['t']))

    w_unscaled, param = _update_weights(mu, sigma, param, X.shape[1], new_feature)

    # scale weights to [0,1] because pystreamfs considers absolute weights for feature selection
    w = MinMaxScaler().fit_transform(w_unscaled.reshape(-1, 1)).flatten()

    return w, param


class _Net(nn.Module):
    def __init__(self, d_in, h, d_out):
        super(_Net, self).__init__()
        self.linear1 = nn.Linear(d_in, h, bias=False)  # define input to hidden layer
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(h, d_out, bias=False)  # define hidden to output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_linear = self.linear1(x)
        h_activation = self.softplus(h_linear)
        out_linear = self.linear2(h_activation)
        y_pred = self.sigmoid(out_linear)
        return y_pred

    def init_weights(self, w_1, w_2):
        # initialize weights with sampled theta
        self.linear1.weight = nn.Parameter(w_1)
        self.linear2.weight = nn.Parameter(w_2)


def _monte_carlo_sampling(in_size, out_size, mu_1, mu_2, sigma_1, sigma_2, param):
    theta_1 = dict()
    theta_2 = dict()
    r_1 = dict()  # reparametrization parameter
    r_2 = dict()

    for l in range(param['L']):
        # Xavier weight initialization TODO: check if correct!
        r_1[l] = torch.distributions.normal.Normal(0, np.sqrt(2/(in_size + out_size))).sample((param['h'], in_size))
        r_2[l] = torch.distributions.normal.Normal(0, np.sqrt(2/(in_size + out_size))).sample((out_size, param['h']))

        theta_1[l] = sigma_1 * r_1[l] + mu_1
        theta_2[l] = sigma_2 * r_2[l] + mu_2

    return theta_1, theta_2, r_1, r_2


def _update_weights(mu, sigma, param, feature_dim, new_feature):
    """
    Compute feature weights as a measure of expected importance and uncertainty

    :param np.ndarray mu: current mu for all features
    :param np.ndarray sigma: current sigma for all features
    :param dict param: parameters
    :param int feature_dim: dimension of the feature space
    :return: w (updated weights), param
    :rtype: np.ndarray, dict
    """
    if 'lambda' not in param:  # initialize lambda and w
        param['lambda'] = 1
        param['w'] = np.zeros(feature_dim)

    # Detect new features
    if new_feature:
        param['w'] = np.append(param['w'], 0)

    lamb = param['lambda']
    w = param['w']

    l1_norm = np.sum(np.abs(mu - lamb * sigma**2))

    # compute weights
    w += param['lr_w'] * (mu - lamb * sigma ** 2) / l1_norm
    param['w'] = w

    # update lambda
    lamb += param['lr_lambda'] * (-l1_norm * np.dot(w, sigma ** 2) - np.sum(np.abs(sigma ** 2)) * (
            np.dot(w, mu) - lamb * np.dot(w, sigma ** 2))) / l1_norm ** 2
    param['lambda'] = lamb

    return w, param
