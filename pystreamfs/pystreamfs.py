import numpy as np
import psutil
import os
import warnings
import time
from pystreamfs.utils import fscr_score, classify
from pystreamfs.plots import plot


def prepare_data(data, target, shuffle):
    """Extract the target and features

    :param numpy.nparray data: dataset
    :param int target: index of the target variable
    :param bool shuffle: set to True if you want to sort the dataset randomly
    :return: X (containing the features), Y (containing the target variable)
    :rtype: numpy.nparray, numpy.nparray
    """

    if shuffle:
        np.random.shuffle(data)

    Y = data[:, target]
    X = np.delete(data, target, 1)

    return X, Y


def simulate_stream(X, Y, fs_algorithm, model, param):
    """Feature selection on simulated data stream

    Stream simulation by batch-wise iteration over dataset.
    Feature selection, classification and saving of performance metrics for every batch

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param function fs_algorithm: feature selection algorithm
    :param object model: Machine learning model for classification
    :param dict param: parameters
    :return: ftr_weights (selected features and their weights over time), stats (performance metrics over time)
    :rtype: numpy.ndarray, dict
    """

    # Do not display warnings in the console
    warnings.filterwarnings("ignore")

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array
    stats = {'time_measures': [],
             'memory_measures': [],
             'acc_measures': [],
             'features': [],
             'fscr_measures': [],
             'time_avg': 0,
             'memory_avg': 0,
             'acc_avg': 0,
             'fscr_avg': 0}

    # Stream simulation
    for i in range(0, X.shape[0], param['batch_size']):
        if fs_algorithm is None or model is None:
            print('Feature selection algorithm or ML model is not defined!')
            return ftr_weights, stats

        # Time taking
        start_t = time.perf_counter()

        # Perform feature selection
        ftr_weights, param = fs_algorithm(X=X[i:i + param['batch_size']], Y=Y[i:i + param['batch_size']],
                                          w=ftr_weights, param=param)
        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]  # top m features

        # Memory and time taking
        t = time.perf_counter() - start_t
        m = psutil.Process(os.getpid()).memory_full_info().uss

        # Classify samples
        model, acc = classify(X, Y, i, selected_ftr, model, param)

        # Save statistics
        stats['time_measures'].append(t)
        stats['memory_measures'].append(m)

        stats['features'].append(selected_ftr.tolist())
        stats['acc_measures'].append(acc)

        # fscr for t >=1
        t = i / param['batch_size']
        if t >= 1:
            fscr = fscr_score(stats['features'][-2], selected_ftr, param['num_features'])
            stats['fscr_measures'].append(fscr)

    # end of stream simulation

    # Compute average statistics
    stats['time_avg'] = np.mean(stats['time_measures'])  # average time in seconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average memory usage in Byte
    stats['acc_avg'] = np.mean(stats['acc_measures'])  # average accuracy score
    stats['fscr_avg'] = np.mean(stats['fscr_measures'])  # average feature selection change rate

    return stats


def plot_stats(stats, ftr_names):
    """Print statistics

    Prints performance metrics obtained during feature selection on simulated data stream

    :param dict stats: statistics
    :param np.array ftr_names: names of original features
    :return: chart
    :rtype: plt.figure
    """

    plot_data = dict()

    # Feature names
    plot_data['ftr_names'] = ftr_names

    # Time in ms
    plot_data['x_time'] = np.array(range(0, len(stats['time_measures'])))
    plot_data['y_time'] = np.array(stats['time_measures']) * 1000
    plot_data['avg_time'] = stats['time_avg'] * 1000

    # Memory in kB
    plot_data['x_mem'] = np.array(range(0, len(stats['memory_measures'])))
    plot_data['y_mem'] = np.array(stats['memory_measures']) / 1000
    plot_data['avg_mem'] = stats['memory_avg'] / 1000

    # Accuracy in %
    plot_data['x_acc'] = np.array(range(0, len(stats['acc_measures'])))
    plot_data['y_acc'] = np.array(stats['acc_measures']) * 100
    plot_data['avg_acc'] = stats['acc_avg'] * 100
    plot_data['q1_acc'] = np.percentile(stats['acc_measures'], 25, axis=0) * 100
    plot_data['q3_acc'] = np.percentile(stats['acc_measures'], 75, axis=0) * 100

    # Selected features
    plot_data['selected_ftr'] = stats['features']

    # FSCR in %
    plot_data['x_fscr'] = np.array(range(1, len(stats['fscr_measures']) + 1))
    plot_data['y_fscr'] = np.array(stats['fscr_measures'])
    plot_data['avg_fscr'] = stats['fscr_avg']

    # Set ticks
    # X ticks
    plot_data['x_ticks'] = np.arange(0, plot_data['x_time'].shape[0], 1)
    if plot_data['x_time'].shape[0] > 30:  # plot every 5th x tick
        plot_data['x_ticks'] = ['' if i % 5 != 0 else b for i, b in enumerate(plot_data['x_ticks'])]

    # Y ticks for selected features
    plot_data['y_ticks_ftr'] = range(0, len(plot_data['ftr_names']))

    chart = plot(plot_data)

    return chart
