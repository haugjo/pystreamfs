import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


def run_ubfs(X, Y, param, **kw):
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
        param['sigma'] = np.ones(m) * param['init_sigma']

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


def _update_weights(mu, sigma, param, feature_dim):
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

    lamb = param['lambda']
    w = param['w']

    # relative mu and sigma
    mu /= np.sum(np.abs(mu))
    sigma /= np.sum(np.abs(sigma))

    # weight computation
    w_update = np.abs(mu) - 2 * lamb * (w * sigma ** 2) - 2 * w
    w += param['lr_w'] * w_update
    param['w'] = w

    lamb_update = -np.dot(w ** 2, sigma ** 2)
    lamb += param['lr_lambda'] * lamb_update
    param['lambda'] = lamb

    # relative w
    w /= np.sum(np.abs(w))

    return w, param


def _check_concept_drift(mu, sigma, param):
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
    if 'drifts' not in param:
        param['drifts'] = []
        param['drift_ind_mean'] = []

        # Todo: needed only during experiments, can be removed afterwards
        param['mu_measures'] = []
        param['sigma_measures'] = []

    if param['drift_basis'] == 'sigma':
        ind_mean = np.mean(sigma)
    else:
        ind_mean = np.mean(mu)

    param['drift_ind_mean'].append(ind_mean)

    # Todo: needed only during experiments, can be removed afterwards
    param['mu_measures'].append(mu)
    param['sigma_measures'].append(sigma)

    if len(param['drift_ind_mean']) > param['range'] * 2:
        center_t = param['drift_ind_mean'].index(param['drift_ind_mean'][-param['range'] - 1])  # center t = index of central measure for given range

        # indicators for maximum or minimum
        min_indicator_left = np.zeros(param['range'])
        min_indicator_right = np.zeros(param['range'])
        max_indicator_left = np.zeros(param['range'])
        max_indicator_right = np.zeros(param['range'])

        for i in range(1, param['range'] + 1):
            # left range
            if param['drift_ind_mean'][center_t - i] > param['drift_ind_mean'][center_t - i + 1]:
                min_indicator_left[i - 1] = True
            elif param['drift_ind_mean'][center_t - i] < param['drift_ind_mean'][center_t - i + 1]:
                max_indicator_left[i - 1] = True

            # right range
            if param['drift_ind_mean'][center_t + i - 1] < param['drift_ind_mean'][center_t + i]:
                min_indicator_right[i - 1] = True
            elif param['drift_ind_mean'][center_t + i - 1] > param['drift_ind_mean'][center_t + i]:
                max_indicator_right[i - 1] = True

        # check for local minima
        if (all(min_indicator_left) and all(min_indicator_right)) or (all(max_indicator_left) and all(max_indicator_right)):
            param['drifts'].append(center_t + 1)

    return param
