from streamfs import streamfs
import numpy as np
import pandas as pd

# Load german credit score dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('Risk', 1).columns)
data = np.array(data)

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size for one iteration of ofs
param['algorithm'] = 'svm'  # apply KNN classifier to calculate accuracy per time t
param['neighbors'] = 5  # set n_neighbors for KNN

# Extract features and target variable
X, Y = streamfs.prepare_data(data, 0, False)
Y[Y == 0] = -1  # change 0 to -1, required by ofs

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', param)

# Print resulting feature weights
print('Final feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(feature_names[stats['features'][-1]]))

# Print params
print('Statistics for one execution of OFS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}kB'.format(stats['memory_avg'] / 1000))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Print average accuracy
print('Average accuracy: {}%'.format(stats['acc_avg']))

# Print MFCR
print('MFCR: {}'.format(stats['mfcr_measures'][-1]))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()
