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

    ########################################
    # 1. DATA PREPARATION
    ########################################
    # Dimensionality of Y
    output_dim = len(np.unique(Y))

    # Format sample as tensor
    x = torch.from_numpy(X).float()  # format sample as tensor
    y = torch.from_numpy(Y).long()

    # initialize uncertainty parameters for distribution of feature weights theta
    if 'mu_1' not in param and 'sigma_1' not in param:
        param['mu_1'] = torch.zeros((param['h'], x.size()[1]))  # input to hidden layer weights
        param['mu_2'] = torch.zeros((output_dim, param['h']))  # hidden to output layer weights

        param['sigma_1'] = torch.ones((param['h'], x.size()[1]))  # input to hidden layer weights
        param['sigma_2'] = torch.ones((output_dim, param['h']))  # hidden to output layer weights

        param['d'] = x.size()[1]  # initial dimensionality of feature space
        param['classes'] = output_dim  # save all initial classes

    mu_1 = param['mu_1'].clone()
    mu_2 = param['mu_2'].clone()
    sigma_1 = param['sigma_1'].clone()
    sigma_2 = param['sigma_2'].clone()

    ########################################
    # 2. CHANGING FEATURES AND/OR CLASSES
    ########################################
    # Detect new features
    new_features = 0  # number of new features
    if x.size()[1] > param['d']:
        mu_1, sigma_1, param, new_features = _new_input_dim(x.size()[1], mu_1, sigma_1, param)  # init new input nodes
        print('New feature detected at t={}'.format(param['t']))

    # Detect new classes
    if len(y.unique()) > param['classes']:
        mu_2, sigma_2, param = _new_output_dim(output_dim, mu_2, sigma_2, param)  # init new output nodes
        print("New class detected at t={}".format(param['t']))

    ########################################
    # 3. MONTE CARLO SAMPLING
    ########################################
    theta_1, theta_2, r_1, r_2 = _monte_carlo_sampling(x.size()[1], output_dim, mu_1, mu_2, sigma_1, sigma_2, param)

    ########################################
    # 4. TRAINING THE NET
    ########################################
    nabla_theta_1 = dict()
    nabla_theta_2 = dict()

    for l in range(param['L']):  # For all samples L
        # Initialize gradient of theta for current sample l
        nabla_theta_1[l] = torch.zeros(theta_1[l].size())
        nabla_theta_2[l] = torch.zeros(theta_2[l].size())

        # Initialize Neural Net, loss function and optimizer
        model = _Net(x.size()[1], param['h'], param['classes'])
        model.init_weights(theta_1[l].clone(), theta_2[l].clone())  # set weights theta
        criterion = nn.NLLLoss()  # Negative Log Likelihood loss for classification
        optimizer = torch.optim.SGD(model.parameters(), lr=param['lr_model'])  # specify optimizer, here SGD

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

    ########################################
    # 5. COMPUTE GRADIENT ON MU AND SIGMA
    ########################################
    # Initialize the gradients
    nabla_mu_1 = torch.zeros(nabla_theta_1[0].size())
    nabla_mu_2 = torch.zeros(nabla_theta_2[0].size())

    nabla_sigma_1 = torch.zeros(nabla_theta_1[0].size())
    nabla_sigma_2 = torch.zeros(nabla_theta_2[0].size())

    for l in range(param['L']):
        # According to gradients in paper:
        nabla_mu_1 += nabla_theta_1[l]
        nabla_mu_2 += nabla_theta_2[l]

        nabla_sigma_1 += nabla_theta_1[l] * r_1[l]
        nabla_sigma_2 += nabla_theta_2[l] * r_2[l]

    nabla_mu_1 /= param['L']  # average for L samples
    nabla_mu_2 /= param['L']
    nabla_sigma_1 /= param['L']
    nabla_sigma_2 /= param['L']

    ########################################
    # 6. UPDATE MU AND SIGMA
    ########################################
    mu_1 -= param['lr_mu'] * nabla_mu_1
    mu_2 -= param['lr_mu'] * nabla_mu_2
    sigma_1 -= param['lr_sigma'] * nabla_sigma_1
    sigma_2 -= param['lr_sigma'] * nabla_sigma_2

    # minimal value of sigma is 0
    sigma_1[sigma_1 < 0] = 0
    sigma_2[sigma_2 < 0] = 0

    # Update param
    param['mu_1'] = mu_1
    param['mu_2'] = mu_2
    param['sigma_1'] = sigma_1
    param['sigma_2'] = sigma_2

    ########################################
    # 7. COMPUTE FEATURE WEIGHTS
    ########################################
    # Use numpy
    mu_1 = mu_1.numpy()
    mu_2 = mu_2.numpy()
    sigma_1 = sigma_1.numpy()
    sigma_2 = sigma_2.numpy()

    # Average mu and sigma depending on all layers in the neural net
    # Goal: Get mu_j and sigma_j for each feature j
    mu_h = np.mean(np.abs(mu_2), axis=0)
    sigma_h = np.mean(sigma_2, axis=0)
    mu_h_norm = mu_h/np.sum(mu_h)  # relative magnitude of weights in layer 2
    sigma_h_norm = sigma_h/np.sum(sigma_h)

    mu = np.dot(np.abs(mu_1).T, mu_h_norm)  # average layer 1 relative to layer 2
    sigma = np.dot(sigma_1.T, sigma_h_norm)

    w_unscaled, param = _update_weights(mu, sigma, param, X.shape[1], new_features)

    # scale weights to [0,1] because pystreamfs considers absolute weights for feature selection
    w = MinMaxScaler().fit_transform(w_unscaled.reshape(-1, 1)).flatten()

    return w, param


class _Net(nn.Module):
    """
    3-layer neural net:
    - with softplus activation of the hidden nodes
    - with softmax activation of the output nodes
    """
    def __init__(self, d_in, h, d_out):
        super(_Net, self).__init__()
        self.linear1 = nn.Linear(d_in, h, bias=False)  # define input to hidden layer
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(h, d_out, bias=False)  # define hidden to output layer
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        h_linear = self.linear1(x)
        h_activation = self.softplus(h_linear)
        out_linear = self.linear2(h_activation)
        y_pred = self.logsoftmax(out_linear)
        return y_pred

    def init_weights(self, w_1, w_2):
        # initialize weights with sampled theta
        self.linear1.weight = nn.Parameter(w_1)
        self.linear2.weight = nn.Parameter(w_2)


def _monte_carlo_sampling(in_size, out_size, mu_1, mu_2, sigma_1, sigma_2, param):
    """
    Monte Carlo sampling of theta (weights of neural net) with reparameterization trick

    :param int in_size: no. of input nodes
    :param int out_size: no. of output nodes
    :param torch.tensor mu_1: mu of first layer weights
    :param torch.tensor mu_2: mu of second layer weights
    :param torch.tensor sigma_1: sigma of first layer weights
    :param torch.tensor sigma_2: sigma of second layer weights
    :param dict param: parameters, includes:
        - int h: no. of hidden nodes
    :return: theta_1, theta_2 (sampled weights for each L), r_1, r_2 (reparameterization parameters for each L)
    :rtype: dict, dict, dict, dict
    """
    theta_1 = dict()
    theta_2 = dict()
    r_1 = dict()  # reparametrization parameter
    r_2 = dict()

    for l in range(param['L']):
        # Xavier weight initialization
        r_1[l] = torch.distributions.normal.Normal(0, np.sqrt(2/(in_size + out_size))).sample((param['h'], in_size))
        r_2[l] = torch.distributions.normal.Normal(0, np.sqrt(2/(in_size + out_size))).sample((out_size, param['h']))

        theta_1[l] = sigma_1 * r_1[l] + mu_1
        theta_2[l] = sigma_2 * r_2[l] + mu_2

    return theta_1, theta_2, r_1, r_2


def _new_input_dim(new_dim, mu_1, sigma_1, param):
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


def _update_weights(mu, sigma, param, feature_dim, new_features):
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
        param['lambda'] = param['init_lambda']
        param['w'] = np.zeros(feature_dim)

    # Detect new features
    if new_features > 0:
        param['w'] = np.append(param['w'], [np.mean(param['w'])] * new_features)

    lamb = param['lambda']
    w = param['w']

    l1_norm = np.sum(np.abs(np.abs(mu) - lamb * sigma**2))

    # compute weights
    w += param['lr_w'] * (np.abs(mu) - lamb * sigma ** 2) / l1_norm
    param['w'] = w

    # update lambda
    lamb += param['lr_lambda'] * (-l1_norm * np.dot(w, sigma ** 2) - np.sum(np.abs(sigma ** 2)) * (
            np.dot(w, np.abs(mu)) - lamb * np.dot(w, sigma ** 2))) / l1_norm ** 2
    param['lambda'] = lamb

    return w, param
