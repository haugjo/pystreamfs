import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

def run_ubfs(X, Y, w, param):
    """
    Uncertainty Based Feature Selection

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

    np.random.seed(42)

    # initialize uncertainty parameters for distribution of feature weights theta
    if 'mu' not in param and 'sigma' not in param:
        m = X.shape[1]  # number of features
        param['mu'] = np.zeros(m)
        param['sigma'] = np.ones(m) * 10
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

    # Update weights
    w, param = _update_weights(w, mu, sigma, param, X.shape[1])

    # concept drift detection
    if param['check_drift'] is True:
        param = _check_concept_drift(mu, sigma, X, Y, param)

    return w, param


def _check_concept_drift(mu, sigma, X, Y, param):
    """
    Check for concept drift in the data based on a combined threshold
    on loss difference and average change in mu

    :param np.ndarray mu: current mu for all features
    :param np.ndarray sigma: current sigma for all features
    :param np.ndarray X: data batch
    :param np.ndarray Y: labels for data batch
    :param dict param: parameters
    :return: param
    :rtype dict
    """
    if 'drift_score' not in param:
        param['drift_score'] = 0
        param['drifts'] = []  # list of times t where drift was detected

    if 'old_mu' in param:
        # Calculate error with current and prior model parameters
        error = _compute_error(X, Y, mu, sigma)  # error with current parameters
        old_error = _compute_error(X, Y, param['old_mu'], param['old_sigma'])  # los with prior parameters

        # Compute difference of expected values with last t
        change_mu = np.abs(np.mean(param['old_mu']) - np.mean(mu))

        if np.abs(error-old_error) > param['drift_error_thr'] and change_mu > param['drift_mu_thr']:
            param['drift_score'] += 1
        else:
            param['drift_score'] = 0  # max(param['drift_score'] - 1, 0)

    param['old_mu'] = mu
    param['old_sigma'] = mu

    # Drift
    if param['drift_score'] == param['drift_count']:
        param['drift_score'] = 0  # set drift score back to 0
        param['drifts'].append(param['t'])

    return param


def _compute_error(X, Y, mu, sigma):
    """
    Compute the mean squared error (MSE) for the given data and feature parameters,
    using the marginal distribution function for computing y_hat.

    :param np.ndarray X: data batch
    :param np.ndarray Y: labels for data batch
    :param np.ndarray mu: current mu for all features
    :param np.ndarray sigma: current sigma for all features
    :return: MSE
    :rtype float
    """

    # dot product x . mu
    dot_x_mu = np.dot(X, mu)

    # dot product x^2 . sigma^2
    dot_x_sigma = np.dot(X ** 2, sigma ** 2)

    # inference -> Eq. 14
    prob_y = norm.cdf(dot_x_mu / np.sqrt(1 + dot_x_sigma))  # prob(y=1)

    return mean_squared_error(Y, prob_y)  # log_loss(Y, prob_y)  # Log loss


def _update_weights(w, mu, sigma, param, feature_dim):
    if 'alpha' not in param:
        param['alpha'] = 0  # initialize parameters
        param['lambda'] = 0
        w = np.zeros(feature_dim)

    alpha = param['alpha']
    lamb = param['lambda']

    alpha = alpha - np.sum(w)
    lamb = lamb - 0.5 * np.dot(w**2, sigma**2)
    w = w + mu - lamb * w * sigma**2 - alpha

    param['alpha'] = alpha
    param['lambda'] = lamb

    return w, param
