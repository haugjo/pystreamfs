from streamfs import streamfs
import numpy as np
import pandas as pd

# Load german credit score dataset
'''
credit_data = pd.read_csv('../datasets/cleaned_german_credit_score.csv')
credit_feature_names = np.array(credit_data.drop('Risk', 1).columns)
credit_data = np.array(credit_data)
'''
har_data = pd.read_csv('../datasets/human_activity_recognition.csv')
har_feature_names = np.array(har_data.drop('Activity', 1).columns)
har_data = har_data[np.isin(har_data['Activity'], ['STANDING', 'WALKING'])]
har_data['Activity'], _ = pd.factorize(har_data['Activity'])
har_data = np.array(har_data)


# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size for one iteration of ofs
param['algorithm'] = 'knn'  # apply KNN classifier to calculate accuracy per time t
param['neighbors'] = 5  # set n_neighbors for KNN

# Original parameters from paper
param['max_n'] = 100  # maximum number of saved instances per cluster
param['e_threshold'] = 3  # error threshold for splitting of a cluster

# Additional parameters
param['boundary_var_multiplier'] = 2  # multiplier for the var. boundary of the closest centroid (run_mcnn())
param['p_diff_threshold'] = 50  # threshold of perc. diff. for split/death rate when drift is assumed (_detect_drift())

# Extract features and target variable
X, Y = streamfs.prepare_data(har_data, 5, False)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'mcnn', param)

# Print resulting feature weights
print('Final feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(har_feature_names[stats['features'][-1]]))

# Print params
print('Statistics for one execution of MCNN with a batch size of {}:'.format(param['batch_size']))

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
streamfs.plot_stats(stats, har_feature_names).show()
