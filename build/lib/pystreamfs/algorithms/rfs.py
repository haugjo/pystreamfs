import numpy as np
from scipy.stats import norm


def run_rfs(X, Y, param, **kw):
    """
    Robust feature selection (working title) based on the calculations by Klaus Broelemann and Gjergji Kasneci

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

    old_mu = param['mu'].copy()
    old_sigma = param['sigma'].copy()
    mu = old_mu.copy()
    sigma = old_sigma.copy()

    for epoch in range(param['epochs']):
        # sort X randomly
        np.random.shuffle(X)

        for i in range(0, X.shape[0], param['mini_batch_size']):
            # Load mini batch
            x = X[i:i+param['mini_batch_size']]
            y = Y[i:i+param['mini_batch_size']]

            # helper functions
            # dot product x . mu
            dot_x_mu = np.dot(x, mu)

            # dot product x^2 . sigma^2
            dot_x_sigma = np.dot(x ** 2, sigma ** 2)

            # inference -> Eq. 14 (not required for feature selection)
            # prob_y = norm.cdf((-1)**(1-y) * ((dot_x_my + bias[0])/np.sqrt(1+dot_x_sigma + bias[1]**2)))
            # print('Error = ', np.abs(y-prob_y))

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

    # compute feature weights
    del_sigma_old = sigma - old_sigma  # penalize increase of uncertainty
    del_sigma_1 = sigma - np.ones(sigma.shape[0])  # penalize absolute uncertainty, relative to standard normal dist.
    w = np.abs(mu) - param['alpha'] * del_sigma_old - param['beta'] * del_sigma_1

    return w, param
