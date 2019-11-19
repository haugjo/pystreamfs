import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Visualizer:
    """
    check: https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
    """
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
                                                                                               y_label='Time (ms)',
                                                                                               title='Feature Selection Time',
                                                                                               palette=self.palette)

        # Predictive Model Training Time
        self.training_time_ax = self.fig.add_subplot(self.grid1[1, 1])
        self.training_time_ax, self.training_time_measures, self.training_time_mean = self._draw_regular_subplot(self.training_time_ax,
                                                                                                                 data_buffer=data_buffer,
                                                                                                                 x_label='Time $t$',
                                                                                                                 y_label='Training Time (ms)',
                                                                                                                 title='Predictive Model Training Time',
                                                                                                                 palette=self.palette)

        # Visualization parameters
        """
        sns.set_context('paper')
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update({'font.size': 12})  # Todo: think about dynamic font size
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']  # color palette

        self.fig = plt.figure(figsize=(20, 25))  # Main figure
        self.fig.canvas.set_window_title('Pystreamfs')

        # Define grids of the figure
        self.performance_grid = gridspec.GridSpec(6, 2)  # Subgrid for results of preformance measures (comp. time/predictive metric)
        self.performance_grid.update(wspace=0.2, hspace=0.8)

        self.fs_grid = gridspec.GridSpec(6, 2)  # Subgrid for results of feature selection (FS metric)
        self.fs_grid.update(hspace=0.1)

        # Init all subplots
        # Plot Predictor Parameters
        self.param_plot = self.fig.add_subplot(self.performance_grid[0, :])
        self.param_plot = self._draw_param_plot(self.param_plot, data_buffer)

        # Plot FS Time
        self.fs_time_plot = self.fig.add_subplot(self.performance_grid[1, 0])
        self.fs_time_plot, self.fs_time_measures, self.fs_time_mean = self._draw_regular_subplot(self.fs_time_plot,
                                                                                                 data_buffer=data_buffer,
                                                                                                 x_label='Time $t$',
                                                                                                 y_label='Time (ms)',
                                                                                                 title='Feature Selection Time',
                                                                                                 palette=self.palette)
        # Plot Training Time
        self.training_time_plot = self.fig.add_subplot(self.performance_grid[1, 1])
        self.training_time_plot, self.training_time_measures, self.training_time_mean = self._draw_regular_subplot(self.training_time_plot,
                                                                                                                   data_buffer=data_buffer,
                                                                                                                   x_label='Time $t$',
                                                                                                                   y_label='Training Time (ms)',
                                                                                                                   title='Predictive Model Training Time',
                                                                                                                   palette=self.palette)

        # Performance Score
        self.predictive_metric_plot = self.fig.add_subplot(self.performance_grid[2, :])
        self.predictive_metric_plot, self.predictive_metric_measures, self.predictive_metric_mean = self._draw_regular_subplot(self.predictive_metric_plot,
                                                                                                                               data_buffer=data_buffer,
                                                                                                                               x_label='Time $t$',
                                                                                                                               y_label=data_buffer.predictor_metric_name,
                                                                                                                               title='Learning Performance',
                                                                                                                               palette=self.palette)

        # Selected features
        self.selection_subplot = self.fig.add_subplot(self.performance_grid[3:-1, :])
        self.selection_subplot = self._draw_selection_subplot(self.selection_subplot,
                                                              data_buffer=data_buffer,
                                                              palette=self.palette)

        # FS Metric -> fs grid
        self.fs_metric_plot = self.fig.add_subplot(self.fs_grid[5, :])
        self.fs_metric_plot, self.fs_metric_measures, self.fs_metric_mean = self._draw_regular_subplot(self.fs_metric_plot,
                                                                                                       data_buffer=data_buffer,
                                                                                                       x_label='Time $t$',
                                                                                                       y_label=data_buffer.fs_metric_name,
                                                                                                       title=None,
                                                                                                       palette=self.palette)
        """

    def init(self):
        self.fs_time_measures.set_data([], [])
        self.fs_time_mean.set_data([], [])
        self.training_time_measures.set_data([], [])
        self.training_time_mean.set_data([], [])

        return self.fs_time_measures, self.fs_time_mean, self.training_time_measures, self.training_time_mean

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

        return self.fs_time_measures, self.fs_time_mean, self.training_time_measures, self.training_time_mean

    @staticmethod
    def _draw_param_plot(ax, data_buffer):  # Todo move to separate file
        ax.axis('off')
        ax.text(0, 1, 'Evaluation of $' + data_buffer.fs_name + '$ feature selector using a $'
                + data_buffer.predictor_name + '$ as predictive model.', weight='bold', size='xx-large')

        # Draw horizontal line as separator
        ax.axhline(0.95, color='black')

        # General Parameters
        ax.text(0, 0.8, 'General Parameters:', weight='bold')
        ax.text(0, 0.65, 'No. of Selected Features = ' + str(data_buffer.n_selected_ftr) + '/'
                + str(data_buffer.n_total_ftr) + '; Batch-Size = ' + str(data_buffer.batch_size)
                + '; Total No. of Observations = ' + str(data_buffer.max_samples))

        # FS Properties Todo: add properties to data buffer
        """  
        ax.text(0, 0.45, '$' + data_buffer.fs_name + '$-Parameters:', weight='bold')

        y = 0.3  # starting coordinates
        x = 0

        for key, value in self.content['fs_prop'].items():
            if isinstance(value, (int, float, str)):  # only plot scalar values
                ax.text(x, y, key + ' = ' + str(value))
                y -= 0.15

                if y < 0:
                    y = 0.3
                    x += 0.15
        """

        return ax

    @staticmethod
    def _draw_regular_subplot(ax, data_buffer, x_label, y_label, title, palette):
        measures, = ax.plot([], [], color=palette[0])  # measures
        avg, = ax.plot([], [], color=palette[3], ls='--')  # moving average

        # Set x-ticks
        """
        x_ticks = np.arange(round((data_buffer.max_samples - data_buffer.pretrain_size) / data_buffer.batch_size))
        ax.set_xticks(x_ticks)
        if len(x_ticks) > 30:  # plot every 5th x tick
            x_tick_labels = ['' if i % 10 != 0 else b for i, b in enumerate(x_ticks)]
        else:
            x_tick_labels = x_ticks

        ax.set_xticklabels(x_tick_labels)
        """

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
        if hasattr(data_buffer, 'ftr_selection'):
            for i, val in enumerate(data_buffer.ftr_selection):
                for v in val:
                    ax.scatter(i, v, marker='_', color=palette[0])

        # y-ticks
        y_ticks = np.arange(0, data_buffer.n_total_ftr, int(data_buffer.n_total_ftr / 20))

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)

        # Set y-lim slightly above and below feature indices
        ax.set_ylim(y_ticks[0]-0.1, y_ticks[-1]+0.1)

        # Set x-lim and x-ticks
        x_ticks = np.arange(round(data_buffer.max_samples / data_buffer.batch_size - data_buffer.pretrain_size))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([])
        ax.set_xlim(-0.05, x_ticks[-1] + 0.05)

        return ax
