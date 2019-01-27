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
param['batch_size'] = 200  # batch size for one iteration of ofs
param['algorithm'] = 'svm'  # apply KNN classifier to calculate accuracy per time t
param['neighbors'] = 5  # set n_neighbors for KNN

# Original parameters from paper
param['max_n'] = 100  # maximum number of saved instances per cluster
param['e_threshold'] = 3  # error threshold for splitting of a cluster

# Additional parameters
# percentage of variables that can at most be outside of variance boundary before new cluster is created
param['max_out_of_var_bound'] = 0.3
param['p_diff_threshold'] = 50  # threshold of perc. diff. for split/death rate when drift is assumed (_detect_drift())

# Extract features and target variable
X, Y = streamfs.prepare_data(data, 0, False)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'mcnn', param)

# Print resulting feature weights
print('Final feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(feature_names[stats['features'][-1]]))

# Print params
print('Statistics for one execution of MCNN with a batch size of {}:'.format(param['batch_size']))

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
