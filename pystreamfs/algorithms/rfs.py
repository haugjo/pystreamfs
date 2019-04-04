import numpy as np
import pandas as pd
from pystreamfs import pystreamfs
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

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

    # set learning rate
    lr_my = 0.01
    lr_sigma = 0.01

    # number of features
    m = X.shape[1]

    # initialize uncertainty parameters for distribution of feature weights theta
    my = np.zeros(m)
    sigma = np.ones(m)
    bias = np.asarray([0, 1])  # normal distributed bias (constant) -> Todo: update bias as well

    for epoch in range(param['epochs']):
        # sort X randomly
        np.random.shuffle(X)

        batch_start = 0

        for i in range(0, X.shape[0], param['batch_size']):
            # Load mini batch
            x = X[i:i+param['batch_size']]
            y = Y[i:i+param['batch_size']]

            # helper functions
            # dot product x . my
            dot_x_my = np.dot(x, my)

            # dot product x^2 . sigma^2
            dot_x_sigma = np.dot(x ** 2, sigma ** 2)

            # inference -> Eq. 14
            prob_y = norm.cdf((-1)**(1-y) * ((dot_x_my + bias[0])/np.sqrt(1+dot_x_sigma + bias[1]**2)))
            print('Error = ', np.abs(y-prob_y))

            # calculate partial derivatives -> Eq. 15 + 16
            # for 1 sample: del_delmy = norm.pdf(dot_my_x/np.sqrt(1+dot_x_sigma)) * (-1)**(1-y) * (x/np.sqrt(1+dot_x_sigma))
            del_delmy = norm.pdf(dot_x_my / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * (x.T / np.sqrt(1 + dot_x_sigma))

            # for 1 sample: del_delsigma = norm.pdf((-1)**(1-y) * dot_my_x/np.sqrt(1+dot_x_sigma)) * (-1)**(1-y) * (2*x**2*sigma * dot_my_x)/(-2 * np.sqrt(1+dot_x_sigma)**3)
            del_delsigma = norm.pdf((-1) ** (1 - y) * dot_x_my / np.sqrt(1 + dot_x_sigma)) * (-1) ** (1 - y) * ((2 * x ** 2 * sigma).T * dot_x_my) / (-2 * np.sqrt(1 + dot_x_sigma) ** 3)

            # update parameters
            my -= lr_my * np.mean(del_delmy, axis=1)
            sigma -= lr_sigma * np.mean(del_delsigma, axis=1)

            # update batch start
            batch_start += param['batch_size']

    # Todo: select weights according to robustness (standard deviation of the feature distribution)
    # set my as feature weights
    w = my

    return w, param


# Load a dataset
data = pd.read_csv('../../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Normalize data
X = MinMaxScaler().fit_transform(X)

# Set parameters
param = dict()
param['epochs'] = 10
param['batch_size'] = 10

run_rfs(X, Y, None, param)
