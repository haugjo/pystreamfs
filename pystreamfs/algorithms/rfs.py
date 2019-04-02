import numpy as np
import pandas as pd
from pystreamfs import pystreamfs
from scipy.stats import norm

def run_rfs(X, Y, w, param):
    """
    Robust feature selection (working title) based on the calculations by Klaus Broelemann and Gjergji Kasneci

    :param X:
    :param Y:
    :param w:
    :param param:
    :return:
    """

    np.random.seed(42)

    # to capture runtime warnings
    # Todo: remove
    np.seterr(all='raise')

    # set learning rate
    lambda_my = 0.01
    lambda_sigma = 0.01

    # number of features
    m = X.shape[1]

    # initialize uncertainty parameters for distribution of feature weights theta
    my = np.zeros(m)
    sigma = np.ones(m)

    # initialize bias
    # bias = np.random.normal(psi[0, 0], psi[0, 1])

    # initialize feature weights theta using uncertainty parameters
    # theta = np.asarray([np.random.normal(mean, std) for [mean, std] in psi[1:]])

    i = 0  # Todo: remove

    # derivatives
    for x, y in zip(X, Y):
        i += 1  # Todo: remove

        x_dot_my = np.dot(x, my)
        x_dot_sigma = np.sqrt(1 + np.dot(x**2, sigma**2))
        dP_dmy = norm.pdf(x_dot_my / x_dot_sigma) * (-1)**(1-y) * (x / x_dot_sigma)

        dP_dsigma = norm.pdf((-1)**(1-y) * (x_dot_my/x_dot_sigma)) * (-1)**(1-y) * (2*(x**2)*sigma*x_dot_my / (-2)*x_dot_sigma**3)

        my -= lambda_my * dP_dmy
        sigma -= lambda_sigma * dP_dsigma

    print(my)
    print(sigma)

    return w, param


# Load a dataset
data = pd.read_csv('../../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

run_rfs(X, Y, None, None)
