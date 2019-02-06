import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(data):
    plt.figure(figsize=(20, 25))
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(wspace=0.2, hspace=0.8)

    # Time
    _plot_one_chart(plt.subplot(gs1[0, 0]), data['x_time'], data['y_time'], data['avg_time'], None, None, data['x_ticks'],
                    't', 'time (ms)', 'Time consumption')

    # Memory
    _plot_one_chart(plt.subplot(gs1[0, 1]), data['x_mem'], data['y_mem'], data['avg_mem'], None, None, data['x_ticks'],
                    't', 'memory (kB)', 'Memory usage')

    # Accuracy
    _plot_one_chart(plt.subplot(gs1[1, :]), data['x_acc'], data['y_acc'], data['avg_acc'], data['q1_acc'], data['q3_acc'],
                    data['x_ticks'], 't', 'accuracy (%)', 'Accuracy')

    # Selected features
    ax = plt.subplot(gs1[2:-1, :])
    ax.set_title('Selected features')
    ax.set_ylabel('feature')
    ax.set_xticks(np.arange(0, data['x_acc'].shape[0], 1))
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
    gs2 = gridspec.GridSpec(5, 2)
    gs2.update(hspace=0)

    # FSCR
    _plot_one_chart(plt.subplot(gs2[4, :]), data['x_fscr'], data['y_fscr'], data['avg_fscr'], None, None,
                    data['x_ticks'], 't', 'fscr (%)', None, True)

    return plt


def _plot_one_chart(ax, x, y, avg, q1, q3, x_ticks, x_label, y_label, title, starts_at_one=False):
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
