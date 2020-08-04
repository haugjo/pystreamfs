import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from timeit import default_timer as timer


class Visualizer:
    """Visualization of results"""
    def __init__(self, data_buffer):
        # General parameters
        sns.set_context('paper')
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update({'font.size': 12})  # Todo: think about dynamic font size
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        self.fig = plt.figure(figsize=(20, 25))
        self.palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']  # color palette

        # Grid specifications
        self.grid1 = gridspec.GridSpec(6, 2)
        self.grid1.update(wspace=0.2, hspace=0.8)  # Specifications upper part
        self.grid2 = gridspec.GridSpec(6, 2)
        self.grid2.update(hspace=0.1)  # Specifications lower part

        # Evaluation parameters -> not updated
        self.param_ax = self.fig.add_subplot(self.grid1[0, :])
        self.param_ax = self._draw_param_plot(self.param_ax, data_buffer)

        # Feature Selection computation time
        self.fs_time_ax = self.fig.add_subplot(self.grid1[1, 0])
        self.fs_time_ax, self.fs_time_measures, self.fs_time_mean = self._draw_regular_subplot(self.fs_time_ax,
                                                                                               data_buffer=data_buffer,
                                                                                               x_label='Time $t$',
                                                                                               y_label='Time (s)',
                                                                                               title='Feature Selection Time',
                                                                                               palette=self.palette)

        # Predictive Model Training Time
        self.training_time_ax = self.fig.add_subplot(self.grid1[1, 1])
        self.training_time_ax, self.training_time_measures, self.training_time_mean = self._draw_regular_subplot(self.training_time_ax,
                                                                                                                 data_buffer=data_buffer,
                                                                                                                 x_label='Time $t$',
                                                                                                                 y_label='Training Time (s)',
                                                                                                                 title='Predictive Model Training Time',
                                                                                                                 palette=self.palette)

        # Performance Score
        self.predictor_metric_ax = self.fig.add_subplot(self.grid1[2, :])
        self.predictor_metric_ax, self.predictor_metric_measures, self.predictor_metric_mean = self._draw_regular_subplot(self.predictor_metric_ax,
                                                                                                                          data_buffer=data_buffer,
                                                                                                                          x_label='Time $t$',
                                                                                                                          y_label=data_buffer.predictor_metric_name,
                                                                                                                          title='Learning Performance',
                                                                                                                          palette=self.palette)

        # Selected features
        self.selection_ax = self.fig.add_subplot(self.grid1[3:-1, :])
        self.selection_ax, self.selection_measures = self._draw_selection_subplot(self.selection_ax,
                                                                                  data_buffer=data_buffer,
                                                                                  palette=self.palette)

        # FS Metric -> grid2
        self.fs_metric_ax = self.fig.add_subplot(self.grid2[5, :])
        self.fs_metric_ax, self.fs_metric_measures, self.fs_metric_mean = self._draw_regular_subplot(self.fs_metric_ax,
                                                                                                     data_buffer=data_buffer,
                                                                                                     x_label='Time $t$',
                                                                                                     y_label=data_buffer.fs_metric_name,
                                                                                                     title=None,
                                                                                                     palette=self.palette)

    def init(self):
        self.fs_time_measures.set_data([], [])
        self.fs_time_mean.set_data([], [])
        self.training_time_measures.set_data([], [])
        self.training_time_mean.set_data([], [])
        self.predictor_metric_measures.set_data([], [])
        self.predictor_metric_mean.set_data([], [])
        self.fs_metric_measures.set_data([], [])
        self.fs_metric_mean.set_data([], [])
        self.selection_measures.set_offsets([])

        return [self.fs_time_measures, self.fs_time_mean, self.training_time_measures, self.training_time_mean,
                self.predictor_metric_measures, self.predictor_metric_mean, self.fs_metric_measures,
                self.fs_metric_mean, self.selection_measures]

    def func(self, data_buffer):
        """function to update plots given the new frames (data_buffer)"""

        x = np.arange(len(data_buffer.fs_time_measures))

        # Feature Selection computation time
        self.fs_time_ax.set_ylim(0, max(data_buffer.fs_time_measures))  # update y-lim
        self.fs_time_measures.set_data(x, data_buffer.fs_time_measures)
        self.fs_time_mean.set_data(x, data_buffer.fs_time_mean)

        # Predictive Model Training Time
        self.training_time_ax.set_ylim(0, max(data_buffer.train_time_measures))
        self.training_time_measures.set_data(x, data_buffer.train_time_measures)
        self.training_time_mean.set_data(x, data_buffer.train_time_mean)

        # Predictor Metric
        if max(data_buffer.predictor_metric_measures) > 1:  # update y-lim if range not in [0,1]
            self.predictor_metric_ax.set_ylim(0, max(data_buffer.predictor_metric_measures))
        self.predictor_metric_measures.set_data(x, data_buffer.predictor_metric_measures)
        self.predictor_metric_mean.set_data(x, data_buffer.predictor_metric_mean)

        # FS Metric
        if max(data_buffer.fs_metric_measures) > 1:  # update y-lim if range not in [0,1]
            self.fs_metric_ax.set_ylim(0, max(data_buffer.fs_metric_measures))
        self.fs_metric_measures.set_data(x, data_buffer.fs_metric_measures)
        self.fs_metric_mean.set_data(x, data_buffer.fs_metric_mean)

        # Selected features
        x = []
        y = []
        for i, val in enumerate(data_buffer.ftr_selection):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        self.selection_measures.set_offsets(list(zip(x, y)))

        return [self.fs_time_measures, self.fs_time_mean, self.training_time_measures, self.training_time_mean,
                self.predictor_metric_measures, self.predictor_metric_mean, self.fs_metric_measures,
                self.fs_metric_mean, self.selection_measures]

    @staticmethod
    def data_generator(evaluator):
        """Regular test_then_train function but yielding a new frame for the live visualization at every iteration"""
        print('Evaluating...')
        while ((evaluator.global_sample_count < evaluator.max_samples) & (timer() - evaluator.start_time < evaluator.max_time)
               & (evaluator.stream.has_more_samples())):
            try:
                evaluator.one_training_iteration()
                yield evaluator.data_buffer  # yield new frame
            except BaseException as exc:
                print(exc)
                break

    @staticmethod
    def _draw_param_plot(ax, data_buffer):  # Todo move to separate file
        ax.axis('off')
        ax.text(0, 1, 'Pystreamfs Evaluation: $' + data_buffer.fs_name + '$ (Feature Selector) --- $'
                + data_buffer.predictor_name + '$ (Predictive Model)', weight='bold', size='xx-large')

        # Draw horizontal line as separator
        ax.axhline(0.95, color='black')

        # General Parameters
        ax.text(0, 0.65, 'n_selected_ftr = ' + str(data_buffer.n_selected_ftr) + '/' + str(data_buffer.n_total_ftr))
        ax.text(0.2, 0.65, 'batch_size = ' + str(data_buffer.batch_size))
        ax.text(0, 0.35, 'samples = ' + str(data_buffer.max_samples))
        ax.text(0.2, 0.35, 'pretrain_size = ' + str(data_buffer.pretrain_size))

        return ax

    @staticmethod
    def _draw_regular_subplot(ax, data_buffer, x_label, y_label, title, palette):
        measures, = ax.plot([], [], color=palette[0])  # measures
        avg, = ax.plot([], [], color=palette[3], ls='--')  # moving average

        # Set x-lim
        x_lim = (data_buffer.max_samples - data_buffer.pretrain_size)/data_buffer.batch_size
        ax.set_xlim(0, x_lim)

        # Set y-lim
        ax.set_ylim(0, 1)

        # Set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.legend(['measures', 'mean'],
                  frameon=True, loc='lower right',
                  bbox_to_anchor=(1, 0.95),
                  fontsize='medium',
                  borderpad=0.2,
                  columnspacing=0.5,
                  ncol=4,
                  handletextpad=0.05,
                  markerscale=0.1)
        ax.set_title(title, weight='bold', loc='left')

        return ax, measures, avg

    @staticmethod
    def _draw_selection_subplot(ax, data_buffer, palette):
        ax.set_title('Selected Features ($m=' + str(data_buffer.n_selected_ftr) + '$) & ' + data_buffer.fs_metric_name, weight='bold')
        ax.set_ylabel('Feature Index')

        # plot selected features for each time step
        measures = ax.scatter([], [], marker='_', color=palette[0])

        # Set x-lim
        x_lim = (data_buffer.max_samples - data_buffer.pretrain_size) / data_buffer.batch_size
        ax.set_xlim(0, x_lim)

        # Set y-lim
        ax.set_ylim(0, data_buffer.n_total_ftr)

        return ax, measures
