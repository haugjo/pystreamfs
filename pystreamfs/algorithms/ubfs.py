import numpy as np
from scipy.stats import norm
from sklearn.metrics import log_loss, mean_squared_error

def run_ubfs(X, Y, param, **kw):
    """
    Uncertainty Based Feature Selection

    :param numpy.nparray X: current data batch
    :param numpy.nparray Y: labels of current batch
    :param dict param: parameters, this includes...
        int epochs: number of epochs (iterations over X)
        int mini_batch_size: no. of samples per mini_batch
        float lr_my: learning rate for the update of mean
        float lr_sigma: learning rate for the update of standard deviation
    :return: w (feature weights), param
    :rtype numpy.ndarray, dict

    ...Todo: add to README.md
    """

    np.random.seed(42)

    # initialize uncertainty parameters for distribution of feature weights theta
    if 'mu' not in param and 'sigma' not in param:
        m = X.shape[1]  # number of features
        param['mu'] = np.zeros(m)
        param['sigma'] = np.ones(m)
        # bias = np.asarray([0, 1])  # normal distributed bias (constant) -> Todo: update bias as well

    mu = param['mu'].copy()
    sigma = param['sigma'].copy()

    for epoch in range(param['epochs']):
        # sort X randomly
        np.random.shuffle(X)

        for i in range(0, X.shape[0], param['mini_batch_size']):
            # Load mini batch
            x = X[i:i+param['mini_batch_size']]
            y = Y[i:i+param['mini_batch_size']]

            # helper functions
            dot_x_mu = np.dot(x, mu)  # dot product x . mu
            dot_x_sigma = np.dot(x ** 2, sigma ** 2)  # dot product x^2 . sigma^2

            # calculate partial derivatives -> Eq. 15 + 16
            # for 1 sample: del_delmy = norm.pdf(dot_my_x/np.sqrt(1+dot_x_sigma)) * (-1)**(1-y) * (x/np.sqrt(1+dot_x_sigma))
            nabla_mu = norm.pdf(dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * (x.T / np.sqrt(1 + dot_x_sigma))

            # for 1 sample: del_delsigma = norm.pdf((-1)**(1-y) * dot_my_x/np.sqrt(1+dot_x_sigma)) * (-1)**(1-y) * (2*x**2*sigma * dot_my_x)/(-2 * np.sqrt(1+dot_x_sigma)**3)
            nabla_sigma = norm.pdf((-1) ** (1 - y) * dot_x_mu / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * ((2 * x ** 2 * sigma).T * dot_x_mu) / (-2 * np.sqrt(1 + dot_x_sigma) ** 3)

            # update parameters
            mu -= param['lr_mu'] * np.mean(nabla_mu, axis=1)
            sigma -= param['lr_sigma'] * np.mean(nabla_sigma, axis=1)

    # update param
    param['mu'] = mu
    param['sigma'] = sigma

    # compute regularizer
    r = (sigma - 1)/(param['t'] + 1)  # + 1 because t starts with 0

    # Compute weights
    w = np.maximum(np.abs(mu) - param['alpha'] * r, np.zeros(mu.shape))

    # concept drift detection
    _check_concept_drift(mu, sigma, X, Y, param)

    return w, param


def _check_concept_drift(mu, sigma, X, Y, param):
    if 'drift_score' not in param:
        param['drift_score'] = 0

    if 'old_mu' in param:
        # Calculate loss with current and prior model parameters
        loss = _compute_loss(X, Y, mu, sigma)  # loss with current parameters
        old_loss = _compute_loss(X, Y, param['old_mu'], param['old_sigma'])  # los with prior parameters

        # Compute difference of expected values with last t
        change_mu = np.abs(np.mean(param['old_mu']) - np.mean(mu))

        if np.abs(loss-old_loss) > 0.025 and change_mu > 0.002:
            param['drift_score'] += 1
        else:
            param['drift_score'] = max(param['drift_score'] - 1, 0)

    param['old_mu'] = mu
    param['old_sigma'] = mu

    # Drift
    if param['drift_score'] == 3:
        param['drift_score'] = 0  # set drift score back to 0
        print('drift at ' + str(param['t']))


def _compute_loss(X, Y, mu, sigma):
    # dot product x . mu
    dot_x_mu = np.dot(X, mu)

    # dot product x^2 . sigma^2
    dot_x_sigma = np.dot(X ** 2, sigma ** 2)

    # inference -> Eq. 14
    prob_y = norm.cdf(dot_x_mu / np.sqrt(1 + dot_x_sigma))  # prob(y=1)

    return mean_squared_error(Y, prob_y)  # log_loss(Y, prob_y)  # Log loss
