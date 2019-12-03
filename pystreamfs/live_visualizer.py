# Used for all the plots in stream_simulator.py for the life visualization


def regular_subplot_mean(ax, x_data, y_data, x_label, y_label, title, mean):
    palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']
    ax.plot(x_data, y_data, color=palette[0])
    ax.plot(x_data, mean, color=palette[3], ls='--')

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(['measures', 'mean'], frameon=True, fontsize='medium',
               borderpad=0.2, columnspacing=0.5, ncol=4, handletextpad=0.05, markerscale=0.1)
    ax.set_title(title, weight='bold')


def regular_subplot(ax, x_data, y_data, x_label, y_label, title):
    palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']
    ax.plot(x_data, y_data, color=palette[0])

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(['measures'], frameon=True, loc='lower right', bbox_to_anchor=(1, 1), fontsize='medium',
              borderpad=0.2, columnspacing=0.5, ncol=4, handletextpad=0.05, markerscale=0.1)
    ax.set_title(title, weight='bold')


def text_subplot(ax, delay):
    ax.axis('off')
    ax.text(0, 1, 'Life visualization', weight='bold', size='xx-large')
    ax.axhline(0.9, color='black')
    ax.text(0, 0.5, 'Update speed: Time taken by the OFS algorithm + time delay of ' + str(delay) + ' s.')
    ax.text(0, 0.1, 'The delay can be changed in the GUI.')

