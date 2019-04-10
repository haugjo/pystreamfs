import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(data):
    """Plot statistics

    :param dict data: statistics prepared for plotting
    :return: plt
    :rtype: plt.figure
    """

    fig = plt.figure(figsize=(20, 25))
    fig.canvas.set_window_title('pystreamfs')
    plt.rcParams.update({'font.size': 12 * data['font_scale']})
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    gs1 = gridspec.GridSpec(6, 2)
    gs1.update(wspace=0.2, hspace=0.8)

    # Parameters
    _plot_parameters(plt.subplot(gs1[0, :]), data)

    # Time
    _plot_one_chart(plt.subplot(gs1[1, 0]), data['x_time'], data['y_time'], data['avg_time'], None, None, data['x_ticks'],
                    't', 'time (ms)', 'Time consumption')

    # Memory
    _plot_one_chart(plt.subplot(gs1[1, 1]), data['x_mem'], data['y_mem'], data['avg_mem'], None, None, data['x_ticks'],
                    't', 'memory (kB)', 'Memory usage')

    # Performance Score
    _plot_one_chart(plt.subplot(gs1[2, :]), data['x_perf'], data['y_perf'], data['avg_perf'], data['q1_perf'], data['q3_perf'],
                    data['x_ticks'], 't', data['metric'], 'Learning Performance')

    # Selected features
    ax = plt.subplot(gs1[3:-1, :])
    ax.set_title('Selected features')
    ax.set_ylabel('feature')
    ax.set_xticks(np.arange(0, data['x_perf'].shape[0], 1))
    ax.set_xticklabels([])

    # plot selected features for each execution
    for i, val in enumerate(data['selected_ftr']):
        for v in val:
            ax.scatter(i, v, marker='_', color='C0')

    if len(data['y_ticks_ftr']) <= 20:
        # if less than 20 features plot tic for each feature and change markers
        ax.set_yticks(data['y_ticks_ftr'])
        ax.set_yticklabels(data['ftr_names'])

    # new grid specs
    gs2 = gridspec.GridSpec(6, 2)
    gs2.update(hspace=0)

    # FSCR
    _plot_one_chart(plt.subplot(gs2[5, :]), data['x_fscr'], data['y_fscr'], data['avg_fscr'], None, None,
                    data['x_ticks'], 't', 'ftr. index change (%)', None, True)

    return plt


def _plot_parameters(ax, data):
    """Plot the header including all parameters

    :param AxesSubplot ax: grid axis
    :param dict data: data including the parameters
    """
    f_size = str(14 * data['font_scale'])
    header_size = str(24 * data['font_scale'])

    ax.axis('off')
    ax.text(0, 1, 'pystreamfs -  Statistics Plot', size=header_size, weight='bold')

    # Header (left)
    ax.text(0, 0.6, 'Feature Selection algorithm:', size=f_size, weight='bold')
    ax.text(0.2, 0.6, data['fs_algorithm'], size=f_size)
    ax.text(0, 0.2, 'Machine Learning model: ', size=f_size, weight='bold')
    ax.text(0.2, 0.2, data['ml_model'], size=f_size)

    # Parameters (right)
    y = 0.4  # starting coordinates
    x = 0.4

    ax.text(0.4, 0.6, 'Parameters:', size=f_size, weight='bold')

    for key, value in data['param'].items():
        if isinstance(value, (int, float, str)):  # only plot scalar values
            ax.text(x, y, key + ' = ' + str(value), size=f_size)
            y -= 0.2

            if y < 0:
                y = 0.4
                x += 0.2


def _plot_one_chart(ax, x, y, avg, q1, q3, x_ticks, x_label, y_label, title, starts_at_one=False):
    """Plot one axis of the chart

    Plots the measurements over time, the mean and optionally an interquartile range

    :param AxesSubplot ax: grid axis
    :param np.ndarray x: x coordinates
    :param np.ndarray y: y coordinates
    :param float avg: mean
    :param float q1: first quartile
    :param float q3: third quartile
    :param list x_ticks: ticks for x-axis
    :param string x_label: x labels
    :param string y_label: y labels
    :param string title: title
    :param boolean starts_at_one: indicates whether plot starts at t=1 (required for fscr score)
    """
    ax.plot(x, y)

    # Mean
    if starts_at_one:
        ax.plot([0, x.shape[0]], [avg, avg])
    else:
        ax.plot([0, x.shape[0] - 1], [avg, avg])

    # Interquartile range
    if q1 is not None:
        ax.fill_between([0, x.shape[0] - 1], q3, q1, facecolor='green', alpha=0.5)

    ax.set_xticks(np.arange(0, x.shape[0] + 1, 1))
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(['measures', 'mean', 'iqr'], loc="best")
