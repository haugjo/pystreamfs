import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

matplotlib.use("TkAgg")


class Visualizer(FuncAnimation):
    """ Live visualization of the evaluation results """
    def __init__(self, evaluator):  # Todo: enable multiple metrics
        # General parameters
        sns.set_context('paper')
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update({'font.size': 12})  # Todo: think about dynamic font size

        self.fig = plt.figure(figsize=(20, 25))
        self.palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']  # color palette

        # Grid specifications
        self.grid1 = gridspec.GridSpec(6, 2)
        self.grid1.update(wspace=0.2, hspace=0.8)  # Specifications upper part
        self.grid2 = gridspec.GridSpec(6, 2)
        self.grid2.update(hspace=0.1)  # Specifications lower part

        # Evaluation parameters -> not updated
        self.param_ax = self.fig.add_subplot(self.grid1[0, :])
        self.param_ax = self._draw_param_plot(self.param_ax, evaluator)

        # Feature Selection computation time
        self.fs_time_ax = self.fig.add_subplot(self.grid1[1, 0])
        self.fs_time_ax, self.fs_time_measures, self.fs_time_mean = self._draw_regular_subplot(self.fs_time_ax,
                                                                                               evaluator=evaluator,
                                                                                               x_label='Time $t$',
                                                                                               y_label='Time (s)',
                                                                                               title='Feature Selection Time',
                                                                                               palette=self.palette)

        # Predictive Model Training Time
        self.training_time_ax = self.fig.add_subplot(self.grid1[1, 1])
        self.training_time_ax, self.training_time_measures, self.training_time_mean = self._draw_regular_subplot(self.training_time_ax,
                                                                                                                 evaluator=evaluator,
                                                                                                                 x_label='Time $t$',
                                                                                                                 y_label='Training Time (s)',
                                                                                                                 title='Predictive Model Training Time',
                                                                                                                 palette=self.palette)

        # Performance Score
        self.predictor_metric_ax = self.fig.add_subplot(self.grid1[2, :])
        self.predictor_metric_ax, self.predictor_metric_measures, self.predictor_metric_mean = self._draw_regular_subplot(self.predictor_metric_ax,
                                                                                                                          evaluator=evaluator,
                                                                                                                          x_label='Time $t$',
                                                                                                                          y_label=evaluator.pred_metrics[0].name,
                                                                                                                          title='Learning Performance',
                                                                                                                          palette=self.palette)

        # Selected features
        self.selection_ax = self.fig.add_subplot(self.grid1[3:-1, :])
        self.selection_ax, self.selection_measures = self._draw_selection_subplot(self.selection_ax,
                                                                                  evaluator=evaluator,
                                                                                  palette=self.palette)

        # FS Metric -> grid2
        self.fs_metric_ax = self.fig.add_subplot(self.grid2[5, :])
        self.fs_metric_ax, self.fs_metric_measures, self.fs_metric_mean = self._draw_regular_subplot(self.fs_metric_ax,
                                                                                                     evaluator=evaluator,
                                                                                                     x_label='Time $t$',
                                                                                                     y_label=evaluator.fs_metrics[0].name,
                                                                                                     title=None,
                                                                                                     palette=self.palette)

        FuncAnimation.__init__(self, self.fig, self.func, frames=self.gen_function(evaluator), init_func=self.init_func, blit=True, repeat=False)

    def init_func(self):
        """ Initialize the subplot placeholders """

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

    def func(self, evaluator):
        """ Update plots given a new frame (current evaluator)

        :param evaluator: (EvaluateFeatureSelection) Evaluator object

        """
        x = np.arange(len(evaluator.feature_selector.comp_time.measures))

        # Feature Selection computation time
        self.fs_time_ax.set_ylim(0, max(evaluator.feature_selector.comp_time.measures))  # update y-lim
        self.fs_time_measures.set_data(x, evaluator.feature_selector.comp_time.measures)
        self.fs_time_mean.set_data(x, evaluator.feature_selector.comp_time.mean)

        # Predictive Model Training Time
        self.training_time_ax.set_ylim(0, max(evaluator.predictor.training_time.measures))
        self.training_time_measures.set_data(x, evaluator.predictor.training_time.measures)
        self.training_time_mean.set_data(x, evaluator.predictor.training_time.mean)

        # Predictor Metric Todo: add iterator for all metrics
        if max(evaluator.pred_metrics[0].measures) > 1:  # update y-lim if range not in [0,1]
            self.predictor_metric_ax.set_ylim(0, max(evaluator.pred_metrics[0].measures))
        self.predictor_metric_measures.set_data(x, evaluator.pred_metrics[0].measures)
        self.predictor_metric_mean.set_data(x, evaluator.pred_metrics[0].mean)

        # FS Metric
        if max(evaluator.fs_metrics[0].measures) > 1:  # update y-lim if range not in [0,1]
            self.fs_metric_ax.set_ylim(0, max(evaluator.fs_metrics[0].measures))
        self.fs_metric_measures.set_data(x, evaluator.fs_metrics[0].measures)
        self.fs_metric_mean.set_data(x, evaluator.fs_metrics[0].mean)

        # Selected features
        x = []
        y = []
        for i, val in enumerate(evaluator.feature_selector.selection):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        self.selection_measures.set_offsets(list(zip(x, y)))

        return [self.fs_time_measures, self.fs_time_mean, self.training_time_measures, self.training_time_mean,
                self.predictor_metric_measures, self.predictor_metric_mean, self.fs_metric_measures,
                self.fs_metric_mean, self.selection_measures]

    @staticmethod
    def gen_function(evaluator):
        """ Yield frames for live plot Todo: check if we can call _test_then_train() directly

        This function corresponds to EvaluateFeatureSelection._test_then_train() but yields a new frame (current evaluator) at every iteration.

        :param evaluator: (EvaluateFeatureSelection) Evaluator object

        """
        while evaluator.global_sample_count < evaluator.max_samples:
            try:
                evaluator.one_training_iteration()
                yield evaluator
            except BaseException as exc:
                print(exc)
                break

    @staticmethod
    def _draw_param_plot(ax, evaluator):
        """ Draw the parameter subplot

        :param ax: (plt.axis) Axis identifier on the grid
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :return: ax
        :rtype plt.axis

        """
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

    @staticmethod
    def _draw_regular_subplot(ax, evaluator, x_label, y_label, title, palette):
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

        measures, = ax.plot([], [], color=palette[0])  # measures
        avg, = ax.plot([], [], color=palette[3], ls='--')  # moving average

        # Set x-lim
        x_lim = (evaluator.max_samples - evaluator.pretrain_size) / evaluator.batch_size
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
    def _draw_selection_subplot(ax, evaluator, palette):
        """ Draw the selected features

        Returns a scatter plot of selected features at each time step

        :param ax: (plt.axis) Axis identifier on the grid
        :param evaluator: (EvaluateFeatureSelection) Evaluator object
        :param palette: (list) Color palette
        :return: ax, measures
        :rtype plt.axis, plt.axis.plot

        """
        ax.set_title('Selected Features ($m=' + str(evaluator.feature_selector.n_selected_ftr) + '$) & ' + evaluator.fs_metrics[0].name, weight='bold')
        ax.set_ylabel('Feature Index')

        # plot selected features for each time step
        measures = ax.scatter([], [], marker='_', color=palette[0])

        # Set x-lim
        x_lim = (evaluator.max_samples - evaluator.pretrain_size) / evaluator.batch_size
        ax.set_xlim(0, x_lim)

        # Set y-lim
        ax.set_ylim(0, evaluator.feature_selector.n_total_ftr)

        return ax, measures
