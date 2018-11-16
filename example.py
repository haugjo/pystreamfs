import streamfs
import numpy as np

# Load german credit score dataset
credit_data = np.genfromtxt('./data/german_credit_score.csv', delimiter=';')

# Define the number of features that you want to return
num_features = 5

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 0)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', num_features)


# Print resulting feature weights
print('Feature weights:\n', w)

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))
