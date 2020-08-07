import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime


class Visualizer:
    """ Live visualization of the evaluation results """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.font_size = 10 * self.evaluator.plot_scale
        self.palette = ['#000000', '#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#91cf60', '#1a9850']
        self.fig = plt.figure('pystreamfs_{}_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), self.evaluator.feature_selector.name.lower()),
                              figsize=(12 * self.evaluator.plot_scale, 8 * self.evaluator.plot_scale))
        self.grid = gridspec.GridSpec(5, 2, wspace=0.2, hspace=0.1, height_ratios=[1/16, 3/12, 3/12, 6/12, 3/12])
        self.grid_low = gridspec.GridSpec(5, 2, wspace=0.2, hspace=0.4, height_ratios=[1/16, 3/12, 3/12, 6/12, 3/12])

        # Initialize subplots
        self.param_ax = self._draw_param_plot(self.grid[0, :], evaluator)

        self.fs_time_ax, self.fs_time_measures, self.fs_time_mean = self._draw_regular_subplot(
            self.grid[1, 0], None, 'Comp. Time (s)', labels=[])

        self.pred_time_ax, self.pred_time_measures, self.pred_time_mean = self._draw_regular_subplot(
            self.grid[1, 1], None, 'Comp. Time (s)', 2, labels=['Training', 'Testing'])

        self.fs_metric_ax, self.fs_metric_measures, self.fs_metric_mean = self._draw_regular_subplot(
            self.grid[2, 0], 'Time Step $t$', 'FS Metric(s)', labels=[metric.name for metric in evaluator.fs_metrics])

        self.pred_metric_ax, self.pred_metric_measures, self.pred_metric_mean = self._draw_regular_subplot(
            self.grid[2, 1], 'Time Step $t$', 'Pred. Metric(s)', len(self.evaluator.fs_metrics), labels=[metric.name for metric in evaluator.pred_metrics])

        self.fs_ax, self.fs_markers = self._draw_selection_subplot(self.grid_low[3, :])

        self.top_features_ax = self._draw_top_features_plot(self.grid_low[4, :])

    def update(self, evaluator):
        """ Update plots given a new frame (current evaluator)

        :param evaluator: (EvaluateFeatureSelection) Evaluator object

        """
        x = np.arange(len(evaluator.feature_selector.comp_time.measures))

        # Line plots
        self._update_lineplot(self.fs_time_ax, x, [evaluator.feature_selector.comp_time], self.fs_time_measures, self.fs_time_mean)

        self._update_lineplot(self.pred_time_ax, x, [evaluator.predictor.training_time, evaluator.predictor.testing_time],
                              self.pred_time_measures, self.pred_time_mean)

        self._update_lineplot(self.pred_metric_ax, x, evaluator.pred_metrics, self.pred_metric_measures, self.pred_metric_mean)

        self._update_lineplot(self.fs_metric_ax, x, evaluator.fs_metrics, self.fs_metric_measures, self.fs_metric_mean)

        # Selected features (scatter)
        x, y = [], []
        for i, val in enumerate(evaluator.feature_selector.selection):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        self.fs_markers.set_offsets(list(zip(x, y)))

        # Top features
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-10:][::-1]
        self.top_features_ax.bar(np.arange(10), counts[top_ftr_idx], color=self.palette[0], width=0.3, zorder=100)
        self.top_features_ax.set_xticklabels(np.asarray(self.evaluator.data_stream.feature_names)[top_ftr_idx])

    @staticmethod
    def _update_lineplot(ax, x, data, plot_measures, plot_mean):
        """ Update line plot

        :param ax: (plt.axis) Axis identifier on the grid
        :param x: (list) X coordinates
        :param data: (list) Array of data
        :param plot_measures: (plt.axis.plot) Plot of measures
        :param plot_mean: (ax.hline) Plot of mean

        """
        # Initialize min and max values
        mn = 0
        mx = -100

        for i in range(len(data)):
            if min(data[i].measures) < mn:  # Update min value
                mn = min(data.measures[i])
            if max(data[i].measures) > mx:  # Update max value
                mx = max(data[i].measures)
                mx += mx * 0.02  # Adjust margins
                mn -= mx * 0.01

            ax.set_ylim(mn, mx)
            plot_measures[i].set_data(x, data[i].measures)
            plot_mean[i].set_data(x, data[i].mean)

    def _draw_param_plot(self, grid, evaluator):
        """ Draw the parameter subplot

        :param grid: (plt.GridSpec) Grid identifier of subplot
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :return: ax
        :rtype plt.axis

        """
        ax = self.fig.add_subplot(grid)
        ax.axis('off')

        # General Parameters
        ax.text(0, 1, 'FS Model: {}'.format(evaluator.feature_selector.name), weight='bold', size=self.font_size)
        ax.text(0, 0.5, 'Selected Features: {}'.format(str(evaluator.feature_selector.n_selected_ftr) + '/' + str(evaluator.feature_selector.n_total_ftr)), size=self.font_size)
        ax.text(0, 0, 'No. of Observations: {}'.format(str(evaluator.max_samples)), size=self.font_size)
        ax.text(0.55, 1, 'Predictive Model: {}'.format(evaluator.predictor.name), weight='bold', size=self.font_size)
        ax.text(0.55, 0.5, 'Batch Size: {}'.format(str(evaluator.batch_size)), size=self.font_size)
        ax.text(0.55, 0, 'Pretrain Size: {}'.format(str(evaluator.pretrain_size)), size=self.font_size)

        return ax

    def _draw_regular_subplot(self, grid, x_label, y_label, num_plots=1, labels=None):
        """  Draw regular subplot (measures + mean)

        :param grid: (plt.GridSpec) Grid identifier of subplot
        :param x_label: (str) Label of x-axis
        :param y_label: (str) Label of y-axis
        :param num_plots: (int) No. of measure/value pairs
        :param labels: (list) Legend labels
        :return: ax, measures, averages
        :rtype plt.axis, plt.axis.plot, plt.axis.plot

        """
        ax = self.fig.add_subplot(grid)
        ax.grid(True)

        # Create empty list of plots
        measures, averages = [], []

        for i in range(num_plots):  # Create plots
            m, = ax.plot([], [], color=self.palette[0 + i])
            measures.append(m)
            avg, = ax.plot([], [], color=self.palette[0 + i], ls='--', label='_nolegend_')
            averages.append(avg)

        # Set x-lim
        x_lim = (self.evaluator.max_samples - self.evaluator.pretrain_size) / self.evaluator.batch_size
        ax.set_xlim(0, x_lim)

        # Set font size
        ax.set_ylabel(y_label, size=self.font_size, labelpad=1.5)
        ax.set_xlabel(x_label, size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.legend(labels, frameon=True, loc='best', fontsize=self.font_size * 0.7, borderpad=0.2, columnspacing=0.5, ncol=4, handletextpad=0.2)

        if x_label is None:  # Remove x-ticks if x_label is None
            ax.set_xticklabels([])

        return ax, measures, averages

    def _draw_selection_subplot(self, grid):
        """ Draw the selected features

        Returns a scatter plot of selected features at each time step

        :param grid: (plt.GridSpec) Grid identifier of subplot
        :return: ax, markers
        :rtype plt.axis, plt.axis.plot

        """
        ax = self.fig.add_subplot(grid)
        ax.grid(True)
        ax.set_ylabel('Feature Index', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)

        # Plot marker for every selected feature
        markers = ax.scatter([], [], marker='.', color=self.palette[0], label='Selected Features', zorder=100)

        # Set axis limits
        ax.set_xlim(-1, (self.evaluator.max_samples - self.evaluator.pretrain_size) / self.evaluator.batch_size + 1)
        ax.set_ylim(-1, self.evaluator.feature_selector.n_total_ftr + 1)

        plt.legend(frameon=True, loc='best', fontsize=self.font_size * 0.7, borderpad=0.2, handletextpad=0.2)

        return ax, markers

    def _draw_top_features_plot(self, grid):
        """ Draw the most selected features over time

        :param grid: (plt.GridSpec) Grid identifier of subplot
        :return: ax
        :rtype plt.axis
        """
        ax = self.fig.add_subplot(grid)
        ax.grid(True, axis='y')
        ax.set_ylabel('Times Selected', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.set_xticks(np.arange(10))
        ax.set_xlim(-0.2, 9.2)

        return ax
