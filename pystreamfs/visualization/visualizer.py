import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Visualizer:
    """ Live visualization of the evaluation results """
    def __init__(self, evaluator, plot_scale=1):
        self.evaluator = evaluator

        # Specify global style parameters
        sns.set_context('paper')
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update({'font.size': 10 * plot_scale})  # Todo: think about dynamic font size

        self.fig = plt.figure(figsize=(12 * plot_scale, 10 * plot_scale))
        self.palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']  # color palette  # Todo: update

        # Grids for subplots
        self.grid1 = gridspec.GridSpec(6, 2)    # Upper region
        self.grid1.update(wspace=0.2, hspace=0.8)
        self.grid2 = gridspec.GridSpec(6, 2)    # Lower region
        self.grid2.update(hspace=0.1)

        # Parameter information subplot
        self.param_ax = self._draw_param_plot(self.grid1[0, :], evaluator)

        # Comp. time features selection
        self.fs_time_ax, self.fs_time_measures, self.fs_time_mean = self._draw_regular_subplot(self.grid1[1, 0], 'Time $t$', 'Time (s)', 'Feature Selection Time')

        # Comp. time prediction (training/testing)
        self.pred_time_ax, self.pred_time_measures, self.pred_time_mean = self._draw_regular_subplot(self.grid1[1, 1], 'Time $t$', 'Training Time (s)', 'Predictive Model Training Time', 2)

        # Predictive metrics
        self.pred_metric_ax, self.pred_metric_measures, self.pred_metric_mean = self._draw_regular_subplot(self.grid1[2, :], 'Time $t$', 'Predictive Metric(s)', 'Learning Performance', len(self.evaluator.fs_metrics))

        # Selected features
        self.fs_ax, self.fs_measures = self._draw_selection_subplot(self.grid1[3:-1, :])

        # FS metrics
        self.fs_metric_ax, self.fs_metric_measures, self.fs_metric_mean = self._draw_regular_subplot(self.grid2[5, :], 'Time $t$', 'FS Metric(s)', None)

    def update(self, evaluator):
        """ Update plots given a new frame (current evaluator)

        :param evaluator: (EvaluateFeatureSelection) Evaluator object

        """
        x = np.arange(len(evaluator.feature_selector.comp_time.measures))

        # Feature Selection computation time
        self.fs_time_ax.set_ylim(0, max(evaluator.feature_selector.comp_time.measures))  # update y-lim
        self.fs_time_measures[0].set_data(x, evaluator.feature_selector.comp_time.measures)
        self.fs_time_mean[0].set_data(x, evaluator.feature_selector.comp_time.mean)

        # Prediction time (training/testing)
        self.pred_time_ax.set_ylim(0, max(evaluator.predictor.training_time.measures))
        self.pred_time_measures[0].set_data(x, evaluator.predictor.training_time.measures)
        self.pred_time_mean[0].set_data(x, evaluator.predictor.training_time.mean)
        self.pred_time_measures[1].set_data(x, evaluator.predictor.testing_time.measures)
        self.pred_time_mean[1].set_data(x, evaluator.predictor.testing_time.mean)

        # Predictive Metrics
        for i, metric in enumerate(evaluator.pred_metrics):
            if max(metric.measures) > 1:  # update y-lim if range not in [0,1]
                self.pred_metric_ax.set_ylim(0, max(metric.measures))
            self.pred_metric_measures[i].set_data(x, metric.measures)
            self.pred_metric_mean[i].set_data(x, metric.mean)

        # FS Metric
        for i, metric in enumerate(evaluator.fs_metrics):
            if max(metric.measures) > 1:  # update y-lim if range not in [0,1]
                self.fs_metric_ax.set_ylim(0, max(metric.measures))
            self.fs_metric_measures[i].set_data(x, metric.measures)
            self.fs_metric_mean[i].set_data(x, metric.mean)

        # Selected features
        x, y = [], []
        for i, val in enumerate(evaluator.feature_selector.selection):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        self.fs_measures.set_offsets(list(zip(x, y)))

    def _draw_param_plot(self, grid, evaluator):
        """ Draw the parameter subplot

        :param ax: (plt.axis) Axis identifier on the grid
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :return: ax
        :rtype plt.axis

        """
        ax = self.fig.add_subplot(grid)

        ax.axis('off')
        ax.text(0, 1, 'Pystreamfs Evaluation: $' + evaluator.feature_selector.name + '$ (Feature Selector) --- $'
                + evaluator.predictor.name + '$ (Predictive Model)', weight='bold', size='xx-large')

        # Draw horizontal line as separator
        ax.axhline(0.95, color='black')

        # General Parameters
        ax.text(0, 0.65, 'n_selected_ftr = ' + str(evaluator.feature_selector.n_selected_ftr) + '/' + str(evaluator.feature_selector.n_total_ftr))
        ax.text(0.2, 0.65, 'batch_size = ' + str(evaluator.batch_size))
        ax.text(0, 0.35, 'samples = ' + str(evaluator.max_samples))
        ax.text(0.2, 0.35, 'pretrain_size = ' + str(evaluator.pretrain_size))

        return ax

    def _draw_regular_subplot(self, grid, x_label, y_label, title, num_plots=1):
        """  Draw a line plot (used for illustrating metrics)

        This returns a two line plots per axis (the measures and a moving average)

        :param ax: (plt.axis) Axis identifier on the grid
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :param x_label: (str) Label of x-axis
        :param y_label: (str) Label of y-axis
        :param title: (str) Plot Title
        :param palette: (list) Color palette
        :return: ax, measures, avg
        :rtype plt.axis, plt.axis.plot, plt.axis.plot

        """

        ax = self.fig.add_subplot(grid)

        measures = []
        averages = []

        for i in range(num_plots):
            m, = ax.plot([], [], color=self.palette[0 + i])  # measures
            measures.append(m)
            avg, = ax.plot([], [], color=self.palette[1 + i], ls='--')  # moving average
            averages.append(avg)

        # Set x-lim
        x_lim = (self.evaluator.max_samples - self.evaluator.pretrain_size) / self.evaluator.batch_size
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

        return ax, measures, averages

    def _draw_selection_subplot(self, grid):
        """ Draw the selected features

        Returns a scatter plot of selected features at each time step

        :param ax: (plt.axis) Axis identifier on the grid
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :param palette: (list) Color palette
        :return: ax, measures
        :rtype plt.axis, plt.axis.plot

        """
        ax = self.fig.add_subplot(grid)

        ax.set_title('Selected Features ($m=' + str(self.evaluator.feature_selector.n_selected_ftr) + '$) & ' + self.evaluator.fs_metrics[0].name, weight='bold')
        ax.set_ylabel('Feature Index')

        # plot selected features for each time step
        measures = ax.scatter([], [], marker='_', color=self.palette[0])

        # Set x-lim
        x_lim = (self.evaluator.max_samples - self.evaluator.pretrain_size) / self.evaluator.batch_size
        ax.set_xlim(0, x_lim)

        # Set y-lim
        ax.set_ylim(0, self.evaluator.feature_selector.n_total_ftr)

        return ax, measures
