import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Visualizer:
    def plot_all_stats(self, stats, ftr_names, fs_algorithm, ml_model, metric, param,
                       font_scale=1):
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

        chart = self._plot(plot_data)

        return chart


    def plot_live_stats(self):
        # Todo
        return 0

    def _plot(self, data):
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
        self._plot_parameters(plt.subplot(gs1[0, :]), data)

        # Time
        self._plot_one_chart(plt.subplot(gs1[1, 0]), data['x_time'], data['y_time'], data['avg_time'], None, None,
                        data['x_ticks'],
                        't', 'time (ms)', 'Time consumption')

        # Memory
        self._plot_one_chart(plt.subplot(gs1[1, 1]), data['x_mem'], data['y_mem'], data['avg_mem'], None, None,
                        data['x_ticks'],
                        't', 'memory (kB)', 'Memory usage')

        # Performance Score
        self._plot_one_chart(plt.subplot(gs1[2, :]), data['x_perf'], data['y_perf'], data['avg_perf'], data['q1_perf'],
                        data['q3_perf'],
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

        # Stability
        self._plot_one_chart(plt.subplot(gs2[5, :]), data['x_stab'], data['y_stab'], data['avg_stab'], None, None,
                        data['x_ticks'], 't', 'Stability', None, True)

        return plt

    def _plot_parameters(self, ax, data):
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
        y = 0.6  # starting coordinates
        x = 0.4

        ax.text(0.4, 0.9, 'Parameters:', size=f_size, weight='bold')

        for key, value in data['param'].items():
            if isinstance(value, (int, float, str)):  # only plot scalar values
                ax.text(x, y, key + ' = ' + str(value), size=f_size)
                y -= 0.2

                if y < -0.1:
                    y = 0.6
                    x += 0.2

    def _plot_one_chart(self, ax, x, y, avg, q1, q3, x_ticks, x_label, y_label, title, starts_at_one=False):
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