import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


class Visualizer:
    def __init__(self, live):
        self.is_live = live
        self.content = dict()
        self.palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']

    def plot_all_stats(self, pipeline):
        ###################################
        # Collect Data
        ###################################
        self.content['feature_names'] = pipeline.feature_names
        self.content['param'] = pipeline.param
        self.content['fs_prop'] = pipeline.feature_selector.prop
        self.content['feature_selector'] = pipeline.feature_selector.name
        self.content['predictor'] = type(pipeline.predictor).__name__
        self.content['metric'] = pipeline.metric.__name__
        self.content['font_scale'] = pipeline.param['font_scale']

        # Time
        self.content['x_time'] = np.arange(0, len(pipeline.stats['time_measures']))
        self.content['y_time'] = np.array(pipeline.stats['time_measures']) * 1000  # in ms
        self.content['avg_time'] = pipeline.stats['time_avg'] * 1000  # in ms

        # Memory
        self.content['x_mem'] = np.arange(0, len(pipeline.stats['memory_measures']))
        self.content['y_mem'] = np.array(pipeline.stats['memory_measures']) / 1000  # in kB
        self.content['avg_mem'] = pipeline.stats['memory_avg'] / 1000

        # Performance Metric
        self.content['x_perf'] = np.arange(0, len(pipeline.stats['perf_measures']))
        self.content['y_perf'] = np.array(pipeline.stats['perf_measures'])
        self.content['avg_perf'] = pipeline.stats['perf_avg']
        self.content['q1_perf'] = np.percentile(pipeline.stats['perf_measures'], 25, axis=0)
        self.content['q3_perf'] = np.percentile(pipeline.stats['perf_measures'], 75, axis=0)

        # Selected Features
        self.content['selected_ftr'] = pipeline.stats['features']

        # Stability
        self.content['x_stab'] = np.arange(1, len(pipeline.stats['stab_measures']) + 1)
        self.content['y_stab'] = np.array(pipeline.stats['stab_measures'])
        self.content['avg_stab'] = pipeline.stats['stab_avg']

        # X ticks
        self.content['x_ticks'] = np.arange(0, self.content['x_time'].shape[0], 1)
        if self.content['x_time'].shape[0] > 30:  # plot every 5th x tick
            self.content['x_ticks'] = ['' if i % 5 != 0 else b for i, b in enumerate(self.content['x_ticks'])]

        # Y ticks for selected features Todo: check if this is needed
        self.content['y_ticks_ftr'] = range(0, len(self.content['feature_names']))

        # Create new plot
        chart = self._create_plot()

        return chart

    def plot_live_stats(self):
        # Todo
        return 0

    def _create_plot(self):
        ###################################
        # Set environment parameters
        ###################################
        sns.set_context('paper')
        plt.style.use('seaborn-darkgrid')
        fig = plt.figure(figsize=(20, 25))
        fig.canvas.set_window_title('pystreamfs')
        plt.rcParams.update({'font.size': 12 * self.content['font_scale']})
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs1 = gridspec.GridSpec(6, 2)
        gs1.update(wspace=0.2, hspace=0.8)

        ###################################
        # First grid: Create Subplots
        ###################################
        # Plot Predictor Parameters
        self._plot_fs_prop(plt.subplot(gs1[0, :]))

        # Plot Time
        self._create_subplot(plt.subplot(gs1[1, 0]), x_data='x_time', y_data='y_time', avg_data='avg_time',
                             x_label='Time t', y_label='Comp. Time (ms)', title='Time Consumption')

        # Plot Memory
        self._create_subplot(plt.subplot(gs1[1, 1]), x_data='x_mem', y_data='y_mem', avg_data='avg_mem',
                             x_label='Time t', y_label='Memory (kB)', title='Memory Usage')

        # Performance Score
        self._create_subplot(plt.subplot(gs1[2, :]), x_data='x_perf', y_data='y_perf', avg_data='avg_perf',
                             x_label='Time t', y_label=self.content['metric'], title='Learning Performance')

        # Selected features
        ax = plt.subplot(gs1[3:-1, :])
        ax.set_title('Selected features')
        ax.set_ylabel('feature')
        ax.set_xticks(np.arange(0, self.content['x_perf'].shape[0], 1))
        ax.set_xticklabels([])

        # plot selected features for each time step
        for i, val in enumerate(self.content['selected_ftr']):
            for v in val:
                ax.scatter(i, v, marker='_', color='C0')

        if len(self.content['y_ticks_ftr']) <= 20:  # Todo: check if this is needed
            # if less than 20 features plot tic for each feature and change markers
            ax.set_yticks(self.content['y_ticks_ftr'])
            ax.set_yticklabels(self.content['feature_names'])

        ###################################
        # Second grid: Create Stability Subplot
        ###################################
        gs2 = gridspec.GridSpec(6, 2)
        gs2.update(hspace=0)

        # Stability
        self._create_subplot(plt.subplot(gs2[5, :]), x_data='x_stab', y_data='y_stab', avg_data='avg_stab',
                             x_label='Feature Selection Stability', y_label=None, title=None, starts_at_one=True)

        return plt

    def _plot_fs_prop(self, ax):
        f_size = str(14 * self.content['font_scale'])
        header_size = str(24 * self.content['font_scale'])

        ax.axis('off')
        ax.text(0, 1, 'pystreamfs -  Statistics Plot', size=header_size, weight='bold')

        # Header (left)
        ax.text(0, 0.6, 'Feature Selection algorithm:', size=f_size, weight='bold')
        ax.text(0.2, 0.6, self.content['feature_selector'], size=f_size)
        ax.text(0, 0.2, 'Machine Learning model: ', size=f_size, weight='bold')
        ax.text(0.2, 0.2, self.content['predictor'], size=f_size)

        # FS Properties (right)
        y = 0.6  # starting coordinates
        x = 0.4

        ax.text(0.4, 0.9, 'Parameters:', size=f_size, weight='bold')

        for key, value in self.content['fs_prop'].items():
            if isinstance(value, (int, float, str)):  # only plot scalar values
                ax.text(x, y, key + ' = ' + str(value), size=f_size)
                y -= 0.2

                if y < -0.1:
                    y = 0.6
                    x += 0.2

    def _create_subplot(self, ax, x_data, y_data, avg_data, x_label, y_label, title, starts_at_one=False):
        ax.plot(self.content[x_data], self.content[y_data])

        # Mean
        if starts_at_one:
            ax.plot([0, self.content[x_data].shape[0]], [self.content[avg_data], self.content[avg_data]])
        else:
            ax.plot([0, self.content[x_data].shape[0] - 1], [self.content[avg_data], self.content[avg_data]])

        # Interquartile range, only for preformance metric
        if x_data == 'x_perf':
            ax.fill_between([0, self.content[x_data].shape[0] - 1], self.content['q3_perf'], self.content['q1_perf'], facecolor='green', alpha=0.5)

        ax.set_xticks(np.arange(0, self.content[x_data].shape[0] + 1, 1))
        ax.set_xticklabels(self.content['x_ticks'])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(['measures', 'mean', 'iqr'], loc="best")