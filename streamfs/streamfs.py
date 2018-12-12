import numpy as np
import psutil
import os

# import FS algorithms
from streamfs.algorithms.ofs import run_ofs
from streamfs.algorithms.fsds import run_fsds


def prepare_data(data, target, shuffle):
    """Extract features X and target variable Y

    :param numpy.nparray data: dataset
    :param int target: index of the target variable
    :param bool shuffle: set to True if you want to sort the dataset randomly

    :return: X (containing the features), Y (containing the target variable)
    :rtype: numpy.nparray, numpy.nparray

    """

    if shuffle:
        np.random.shuffle(data)

    y = data[:, target]  # extract target variable
    x = np.delete(data, target, 1)  # delete target column

    return x, y


def simulate_stream(X, Y, algorithm, param):
    """Apply Feature Selection on stream data

    Iterate over all datapoints in a given matrix to simulate a data stream.
    Perform given feature selection algorithm and return an array containing the weights for each (selected) feature

    :param numpy.ndarray X: dataset
    :param numpy array Y: target
    :param str algorithm: feature selection algorithm
    :param int num_features: number of features that should be returned

    :return: ftr_weights (containing the weights of the (selected) features), stats (contains i.a. average computation
        time in ms and memory usage (in percent of physical memory) for one execution of the fs algorithm
    :rtype: numpy.ndarray, dict

    .. todo: enable OFS for different batch sizes
    """

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array

    stats = {'memory_start': psutil.Process(os.getpid()).memory_percent(),  # get current memory usage of the process
             'time_measures': [],
             'memory_measures': [],
             'time_avg': 0,
             'memory_avg': 0}

    for i in range(0, X.shape[0], param['batch_size']):
        # OFS
        if algorithm == 'ofs':
            if param['batch_size'] == 1:
                ftr_weights, time, memory = run_ofs(X[i], Y[i], ftr_weights, param['num_features'])
            else:
                print('WARNING: OFS currently only works for a batch size of 1!\n')
                return ftr_weights, stats
                # ftr_weights, time, memory = _ofs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], ftr_weights, param['num_features'])

        # FSDS
        elif algorithm == 'fsds':
            x_t = X[i:i+param['batch_size']].T  # transpose x batch because FSDS assumes rows to represent features
            ftr_weights, time, memory, param['b'], param['ell'] = run_fsds(param['b'], x_t, X.shape[1], param['k'], param['ell'])

        # no valid algorithm selected
        else:
            print('Specified feature selection algorithm is not defined!')
            return ftr_weights, stats

        # add difference in memory usage and computation time
        stats['memory_measures'].append(memory - stats['memory_start'])
        stats['time_measures'].append(time)

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average percentage of used memory

    return ftr_weights, stats
