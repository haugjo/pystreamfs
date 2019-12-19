import torch
from torch import nn
from collections import OrderedDict
import numpy as np


def monte_carlo_sampling(mc_samples, mu, sigma, size, input_dim=None, output_dim=None):
    """
    Monte Carlo sampling of theta with reparameterization trick

    size = (rows, columns) of weight matrix theta
    input_dim, output_dim = dimensions of ANN -> only needed for xavier initialization
    """
    theta = dict()
    epsilon = dict()  # reparametrization parameter

    for l in range(mc_samples):
        if input_dim is not None:
            # Xavier weight initialization
            epsilon[l] = torch.distributions.normal.Normal(0, np.sqrt(2 / (input_dim + output_dim))).sample(size)
        else:
            epsilon[l] = torch.distributions.normal.Normal(0, 1).sample(size)
        theta[l] = epsilon[l] * torch.from_numpy(sigma).float() + torch.from_numpy(mu).float()

    return theta, epsilon


class Net(nn.Module):
    """Feed Forward Neural net with fully connected layers
    - with softplus activation of the hidden nodes
    - with softmax activation of the output nodes
    """
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        super(Net, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim, bias=False)  # define input to hidden layer
        self.softplus = nn.Softplus()

        self.linear_hidden = nn.ModuleList()
        self.softplus_hidden = nn.ModuleList()
        for h in range(hidden_layers - 1):  # minus 1 because 1 hidden layer is implemented per se
            self.linear_hidden.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.softplus_hidden.append(nn.Softplus())

        self.linear_out = nn.Linear(hidden_dim, output_dim, bias=False)  # define hidden to output layer
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        input_hidden = self.linear_in(x)
        hidden = self.softplus(input_hidden)

        for layer, activation in zip(self.linear_hidden, self.softplus_hidden):
            hidden = layer(hidden)
            hidden = activation(hidden)

        out_linear = self.linear_out(hidden)
        y_pred = self.logsoftmax(out_linear)
        return y_pred

    def init_weights(self, theta):
        # initialize weights of first layer
        self.linear_in.weight = nn.Parameter(theta)


class SDT(nn.Module):
    """
    Code based on "Distilling a Neural Network Into a Soft Decision Tree" (Frosst, Hinton 2017)
    as provided at https://github.com/AaronX121/Soft-Decision-Tree/blob/master/SDT.py  (with slight changes)
    """
    def __init__(self, depth, lamda, input_dim, output_dim, lr):
        super(SDT, self).__init__()
        self.depth = depth
        self.inner_node_num = 2 ** self.depth - 1
        self.leaf_num = 2 ** self.depth

        # Different penalty coefficients for nodes in different layer
        self.penalty_list = [lamda * (2 ** (-dp)) for dp in range(0, self.depth)]

        # Initialize inner nodes and leaf nodes
        self.inner_nodes = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(input_dim, self.inner_node_num, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self.leaf_nodes = nn.Linear(self.leaf_num, output_dim, bias=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, data):
        _mu, _penalty = self._forward(data)
        output = self.leaf_nodes(_mu)
        return output, _penalty

    """ Core implementation on data forwarding in SDT """
    def _forward(self, data):
        batch_size = data.size()[0]
        path_prob = self.inner_nodes(data)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        _mu = data.data.new(batch_size, 1, 1).fill_(1.)
        _penalty = torch.tensor(0.)

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)  # extract inner nodes in current layer to calculate regularization term
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        mu = _mu.view(batch_size, self.leaf_num)
        return mu, _penalty

    """ Calculate penalty term for inner-nodes in different layer """
    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        penalty = torch.tensor(0.)
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))
        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) / torch.sum(_mu[:, node // 2], dim=0)
            penalty -= self.penalty_list[layer_idx] * 0.5 * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

    """ Add constant 1 onto the front of each instance 
    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1)
        input = torch.cat((bias, input), 1)
        return input
    """


'''
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
'''''''''
