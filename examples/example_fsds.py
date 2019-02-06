from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import fsds

# Load a dataset
data = pd.read_csv('../datasets/har.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Load a FS algorithm
algorithm = fsds.run_fsds

# Define parameters
param = dict()
param['batch_size'] = 50  # batch size for one iteration, must be at least the same size than the no. of clusters!!
param['num_features'] = 5
param['algorithm'] = 'tree'  # classification algorithm: here Decision Tree

param['B'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = 2  # no. of singular values (can be equal to no. of clusters/classes -> here 2 for binary class.)
param['m'] = data.shape[1]-1  # no. of original features

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, algorithm, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names).show()
