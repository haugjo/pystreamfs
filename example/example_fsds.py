from streamfs import streamfs
import numpy as np
import pandas as pd

# Load humane activity recognition
har_data = pd.read_csv('../datasets/human_activity_recognition.csv')
feature_names = np.array(har_data.columns)
har_data = np.array(har_data)

# Extract features and target variable
X, Y = streamfs.prepare_data(har_data, 562, False)
Y, clusters = pd.factorize(Y)
X = np.array(X, dtype='float')  # data has to be provided as float for internal SVD
Y = np.array(Y)

# Define initial parameters for FSDS
param = dict()
param['b'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = clusters.size  # no. of singular values (equal to no. of clusters)
param['batch_size'] = 1000  # batch size for one iteration
param['num_features'] = 10

# Simulate feature selection on a data stream (for the given data, FS algorithm and parameters)
w, stats = streamfs.simulate_stream(X, Y, 'fsds', param)

# Print resulting feature weights
print('Feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(stats['features'][-1]))

# Print params
print('Statistics for one execution of FSDS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()


