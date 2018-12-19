from streamfs import streamfs
import numpy as np
import pandas as pd

# Load german credit score dataset
# Todo: find a dataset with drift
credit_data = pd.read_csv('../datasets/cleaned_german_credit_score.csv')
feature_names = np.array(credit_data.columns)
credit_data = np.array(credit_data)

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 10  # batch size for one iteration of ofs

# Original parameters from paper
param['max_n'] = 100  # maximum number of saved instances per cluster
param['e_threshold'] = 3  # error threshold for splitting of a cluster

# Additional parameters
param['boundary_var_add_coef'] = 2  # factor added to the var. boundary of the closest centroid (run_mcnn()) Todo: change to multiplier
param['p_diff_threshold'] = 50  # threshold of perc. diff. for split/death rate when drift is assumed (_detect_drift())

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 23, False)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'mcnn', param)

# Print resulting feature weights
print('Final feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(feature_names[stats['features'][-1]]))

# Print params
print('Statistics for one execution of MCNN with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg']))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()
