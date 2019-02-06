from pystreamfs import pystreamfs_new
import numpy as np
import pandas as pd
from pystreamfs.algorithms import ofs

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Load a FS algorithm
algorithm = ofs.run_ofs

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size
param['algorithm'] = 'knn'  # classification algorithm: here KNN
param['neighbors'] = 5  # n_neighbors for KNN

# Extract features and target variable
X, Y = pystreamfs_new.prepare_data(data, 0, False)
Y[Y == 0] = -1  # change 0 to -1, required by ofs

# Data stream simulation
w, stats = pystreamfs_new.simulate_stream(X, Y, algorithm, param)

# Plot statistics
pystreamfs_new.plot_stats(stats, feature_names).show()
