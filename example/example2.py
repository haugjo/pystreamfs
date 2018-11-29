import streamfs
import numpy as np
import pandas as pd

from stream_fast_weight import StreamFastWeight
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

# Load humane activity recognition
har_data = np.array(pd.read_csv('./example_data/HAR_train.csv'))

# Define the number of features that you want to return
num_features = 10

# Extract features and target variable
X, Y = streamfs.prepare_data(har_data, 562, False)
Y, uniques = pd.factorize(Y)
X = np.array(X, dtype='float')  # data has to be provided as float for internal SVD
Y = np.array(Y)


# Define initial parameters for FSDS
param = dict()
param['b'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = 5  # no. of singular values (equal to no. of clusters)
param['batch_size'] = 1000  # batch size for one iteration

# Simulate feature selection on a data stream (for the given data, FS algorithm and parameters)
w, stats = streamfs.simulate_stream(X, Y, 'fsds', param)

print('result from streamfs:\n')
# Print resulting feature weights
selected_idx = np.argsort(w)[::-1][:num_features]
print('Feature weights:\n', w[selected_idx])
print('Selected features: {}'.format(selected_idx))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

##############
fsds = StreamFastWeight(X.shape[1], k=uniques.size)


# Huang code:
X = X.T.astype(float)
nt = 1000

for head in range(0, X.shape[1], nt):  # each time step
    scores = fsds.update(X[:, head:head + nt])

# selected top-n features at the final step
selected_idx = np.argsort(scores)[::-1][:num_features]

print('#####################')
print('Results for Huang implementation:')
# Print resulting feature weights
print('Feature weights:\n', scores[selected_idx])
print('Selected features: {}'.format(selected_idx))


