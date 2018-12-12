from streamfs import streamfs
import numpy as np
import pandas as pd

# Load german credit score dataset
credit_data = pd.read_csv('../datasets/german_credit_score.csv', delimiter=';', header=None)
feature_names = np.array(credit_data.columns)
credit_data = np.array(credit_data)

# Define parameters
param = dict()
param['num_features'] = 10  # number of features to return
param['batch_size'] = 1  # batch size for one iteration of ofs

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 0, False)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', param)

# Print resulting feature weights
print('Feature weights:\n', w[stats['features'][-1]])
print('Indices of selected features: {}'.format(stats['features'][-1]))

# Print params
print('Statistics for one execution of OFS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Plot time and memory consumption
streamfs.print_stats(stats, feature_names).show()
