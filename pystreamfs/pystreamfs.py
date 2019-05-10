import numpy as np
import psutil
import os
import warnings
import time
from pystreamfs.utils import nogueira_stability, classify
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


def simulate_stream(X, Y, fs_algorithm, model, metric, param):
    """Feature selection on simulated data stream

    Stream simulation by batch-wise iteration over dataset.
    Feature selection, classification and saving of performance metrics for every batch

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param function fs_algorithm: feature selection algorithm
    :param object model: Machine learning model for classification
    :param function metric: performance metric
    :param dict param: parameters
    :return: ftr_weights (selected features and their weights over time), stats (performance metrics over time)
    :rtype: numpy.ndarray, dict
    """

    # Do not display warnings in the console
    warnings.filterwarnings("ignore")

    ftr_weights = []  # empty feature weights array
    stats = {'time_measures': [],
             'memory_measures': [],
             'perf_measures': [],
             'features': [],
             'stab_measures': [],
             'time_avg': 0,
             'memory_avg': 0,
             'perf_avg': 0,
             'stab_avg': 0}

    # Stream simulation
    for i in range(0, X.shape[0], param['batch_size']):  # data stream
        t = i / param['batch_size']  # time window
        param['t'] = t

        if 'feature_stream' in param and t in param['feature_stream']:  # feature stream
            ftr_indices = param['feature_stream'][t]
        elif 'feature_stream' not in param:
            ftr_indices = np.arange(0, X.shape[1])  # all features are available

        # Time taking
        start_tim = time.perf_counter()

        # Perform feature selection
        ftr_weights, param = fs_algorithm(X=X[i:i + param['batch_size'], ftr_indices], Y=Y[i:i + param['batch_size']],
                                          w=ftr_weights, param=param)
        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]  # top m absolute weights (features with highest influence)

        # Memory and time taking
        tim = time.perf_counter() - start_tim
        mem = psutil.Process(os.getpid()).memory_full_info().uss

        # Classify samples
        model, perf_score = classify(X, Y, i, selected_ftr, model, metric, param)

        # Save statistics
        stats['time_measures'].append(tim)
        stats['memory_measures'].append(mem)

        stats['features'].append(selected_ftr.tolist())
        stats['perf_measures'].append(perf_score)

        # stability measure for t >=1
        if t >= 1:
            stability = nogueira_stability(X.shape[1], stats['features'])
            stats['stab_measures'].append(stability)

    # end of stream simulation

    # Compute average statistics
    stats['time_avg'] = np.mean(stats['time_measures'])  # average time in seconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average memory usage in Byte
    stats['perf_avg'] = np.mean(stats['perf_measures'])  # average performance metric
    stats['stab_avg'] = np.mean(stats['stab_measures'])  # average feature selection change rate

    # Add detections of concept drift
    # Todo: plot drifts + remove temporal parameters before ploting
    if 'check_drift' in param and param['check_drift'] is True:
        stats['drifts'] = param['drifts']

    return stats


def plot_stats(stats, ftr_names, fs_algorithm, ml_model, metric, param, font_scale=1):
    """Print statistics

    Prints performance metrics obtained during feature selection on simulated data stream

    :param dict stats: statistics
    :param np.ndarray ftr_names: names of original features
    :param dict param: parameters
    :param string fs_algorithm: name of the fs algorithm
    :param string ml_model: name of the ML model
    :param string metric: name of the performance metric
    :param float font_scale: factor by which the standard font size for text is scaled
    :return: chart
    :rtype: plt.figure
    """

    plot_data = dict()

    # Feature names & parameters
    plot_data['ftr_names'] = ftr_names
    plot_data['param'] = param
    plot_data['fs_algorithm'] = fs_algorithm
    plot_data['ml_model'] = ml_model
    plot_data['metric'] = metric
    plot_data['font_scale'] = font_scale

    # Time in ms
    plot_data['x_time'] = np.array(range(0, len(stats['time_measures'])))
    plot_data['y_time'] = np.array(stats['time_measures']) * 1000
    plot_data['avg_time'] = stats['time_avg'] * 1000

    # Memory in kB
    plot_data['x_mem'] = np.array(range(0, len(stats['memory_measures'])))
    plot_data['y_mem'] = np.array(stats['memory_measures']) / 1000
    plot_data['avg_mem'] = stats['memory_avg'] / 1000

    # Performance score
    plot_data['x_perf'] = np.array(range(0, len(stats['perf_measures'])))
    plot_data['y_perf'] = np.array(stats['perf_measures'])
    plot_data['avg_perf'] = stats['perf_avg']
    plot_data['q1_perf'] = np.percentile(stats['perf_measures'], 25, axis=0)
    plot_data['q3_perf'] = np.percentile(stats['perf_measures'], 75, axis=0)

    # Selected features
    plot_data['selected_ftr'] = stats['features']

    # Stability
    plot_data['x_stab'] = np.array(range(1, len(stats['stab_measures']) + 1))
    plot_data['y_stab'] = np.array(stats['stab_measures'])
    plot_data['avg_stab'] = stats['stab_avg']

    # Set ticks
    # X ticks
    plot_data['x_ticks'] = np.arange(0, plot_data['x_time'].shape[0], 1)
    if plot_data['x_time'].shape[0] > 30:  # plot every 5th x tick
        plot_data['x_ticks'] = ['' if i % 5 != 0 else b for i, b in enumerate(plot_data['x_ticks'])]

    # Y ticks for selected features
    plot_data['y_ticks_ftr'] = range(0, len(plot_data['ftr_names']))

    chart = plot(plot_data)

    return chart
