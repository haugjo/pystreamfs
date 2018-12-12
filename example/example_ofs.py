from streamfs import streamfs
import numpy as np

# Load german credit score dataset
credit_data = np.genfromtxt('../datasets/german_credit_score.csv', delimiter=';')

# Define parameters
param = dict()
param['num_features'] = 10  # number of features to return
param['batch_size'] = 1  # batch size for one iteration of ofs

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 0, False)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', param)

# Print resulting feature weights
idx = np.nonzero(w)
print('Feature weights:\n', w[idx])
print('Indices of selected features: {}'.format(idx[0]))

# Print params
print('Statistics for one execution of OFS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))
