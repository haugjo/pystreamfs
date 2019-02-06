import numpy as np
import psutil
import os
import warnings
import time
from pystreamfs.utils import comp_fscr, perform_learning
from pystreamfs.plots import plot


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

    Y = data[:, target]
    X = np.delete(data, target, 1)

    return X, Y


def simulate_stream(X, Y, ftr_selection, param):
    """Apply Feature Selection on stream data

    Iterate over all datapoints in a given matrix to simulate a data stream.
    Perform given feature selection algorithm and return an array containing the weights for each (selected) feature

    :param numpy.ndarray X: dataset
    :param numpy array Y: target
    :param function algorithm: feature selection algorithm
    :param dict param: parameters for feature selection

    :return: ftr_weights (containing the weights of the (selected) features), stats (contains i.a. average computation
        time in ms and memory usage (in percent of physical memory) for one execution of the fs algorithm
    :rtype: numpy.ndarray, dict
    """

    # Do not display warnings in the console
    warnings.filterwarnings("ignore")

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array
    classifier = None  # ML model
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
        if ftr_selection is None:
            print('Feature selection algorithm is not defined!')
            return ftr_weights, stats

        # Memory and time taking
        start_t = time.perf_counter()
        start_m = psutil.Process(os.getpid()).memory_full_info().uss

        # Perform feature selection
        ftr_weights, param = ftr_selection(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], ftr_weights, param)
        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]  # top m features

        # Memory and time taking
        t = time.perf_counter() - start_t
        m = psutil.Process(os.getpid()).memory_full_info().uss - start_m

        # Classify samples
        classifier, acc = perform_learning(X, Y, i, selected_ftr, classifier, param)

        # Save statistics
        stats['time_measures'].append(t)
        stats['memory_measures'].append(m)

        stats['features'].append(selected_ftr.tolist())
        stats['acc_measures'].append(acc)

        # fscr for t >=1
        t = i / param['batch_size']
        if t >= 1:
            fscr = comp_fscr(stats['features'][-2], selected_ftr, param['num_features'])
            stats['fscr_measures'].append(fscr)

    # end of stream simulation

    # Compute average statistics
    stats['time_avg'] = np.mean(stats['time_measures'])  # average time in seconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average memory usage in Byte
    stats['acc_avg'] = np.mean(stats['acc_measures'])  # average accuracy score
    stats['fscr_avg'] = np.mean(stats['fscr_measures'])  # average feature selection change rate

    return ftr_weights, stats


def plot_stats(stats, ftr_names):
    """Print Time and Memory consumption

    Print the time, memory and accuracy measures as provided in stats. Also print the average time, memory consumption, accuracy

    :param dict stats: statistics
    :param np.array ftr_names: contains feature names (if included, features will be plotted
    :return: plt (plot containing 2 subplots for time and memory)
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
