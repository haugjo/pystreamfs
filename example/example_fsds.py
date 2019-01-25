from streamfs import streamfs
import numpy as np
import pandas as pd

# Load humane activity recognition
data = pd.read_csv('../datasets/har_binary.csv')
feature_names = np.array(data.drop('walking', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = streamfs.prepare_data(data, 0, False)

# Define initial parameters for FSDS
param = dict()
param['b'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = 2  # no. of singular values (can be equal to no. of clusters/classes -> here always 2 for binary class.)
param['batch_size'] = 50  # batch size for one iteration, must be at least the same size than the no. of clusters!!
param['num_features'] = 5
param['algorithm'] = 'tree'  # apply Classification Tree classifier to calculate accuracy per time t

# Simulate feature selection on a data stream (for the given data, FS algorithm and parameters)
w, stats = streamfs.simulate_stream(X, Y, 'fsds', param)

# Print resulting feature weights
print('Final feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(stats['features'][-1]))

# Print params
print('Statistics for one execution of FSDS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}kB'.format(stats['memory_avg'] / 1000))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Print average accuracy
print('Average accuracy: {}%'.format(stats['acc_avg']))

# Print fscr
print('Average fscr: {}'.format(stats['fscr_avg']))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()


