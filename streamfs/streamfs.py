import numpy as np
import psutil
import os
import matplotlib.pyplot as plt

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
             'features': [],
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

        # save indices of currently selected features
        stats['features'].append(np.argsort(abs(ftr_weights))[::-1][:param['num_features']])

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average percentage of used memory

    return ftr_weights, stats


def print_stats(stats, ftr_names = None):
    """Print Time and Memory consumption

    Print the time and memory measures as provided in stats. Also print the average time and memory consumption

    :param dict stats: statistics
    :param np.array ftr_names: contains feature names (if included, features will be plotted
    :return: plt (plot containing 2 subplots for time and memory)
    """

    # plot time and memory
    x_time = np.array(range(0, len(stats['time_measures'])))
    y_time = np.array(stats['time_measures'])*1000

    x_mem = np.array(range(0, len(stats['memory_measures'])))
    y_mem = np.array(stats['memory_measures'])*100

    plt.figure(figsize=(15, 25))
    plt.subplots_adjust(wspace=0.3)

    plt.subplot2grid((3, 2), (0, 0))
    plt.plot(x_time, y_time)
    plt.plot([0, x_time.shape[0]-1], [stats['time_avg'], stats['time_avg']])
    plt.xlabel('execution no.')
    plt.ylabel('time (ms)')
    plt.title('Time consumption for FS')
    plt.legend(['time measures', 'avg. time'])

    plt.subplot2grid((3, 2), (0, 1))
    plt.plot(x_mem, y_mem)
    plt.plot([0, x_mem.shape[0]-1], [stats['memory_avg'] * 100, stats['memory_avg'] * 100])
    plt.xlabel('execution no.')
    plt.ylabel('memory (% of RAM)')
    plt.title('Memory consumption for FS')
    plt.legend(['memory measures', 'avg. memory'])

    # plot selected features
    ftr_indices = range(0, len(ftr_names))

    plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)
    plt.title('Selected features')
    plt.xlabel('execution no.')
    plt.ylabel('feature')

    # plot selected features for each execution
    for i, val in enumerate(stats['features']):
        for v in val:
            plt.scatter(i, v, marker='_', color='C0')

    # markers indicating final list of features
    marker_y = '_'
    marker_n = '_'

    if len(ftr_indices) <= 30:
        # if less than 30 features plot tic for each feature and change markers
        plt.yticks(ftr_indices, ftr_names)
        marker_y = 'P'
        marker_n = 'x'

    # plot final set of features
    for i in ftr_indices:
        if i in stats['features'][-1]:
            plt.scatter(len(stats['features']), i, marker=marker_y, color="C2")
        else:
            plt.scatter(len(stats['features']), i, marker=marker_n, color="C3")

    return plt
