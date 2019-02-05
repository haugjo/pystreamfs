from pystreamfs import pystreamfs
import numpy as np
import pandas as pd

# Load a dataset
data = pd.read_csv('../datasets/usenet.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 500  # batch size
param['algorithm'] = 'knn'  # classification algorithm: here KNN
param['neighbors'] = 5  # n_neighbors for KNN

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Data stream simulation
w, stats = pystreamfs.simulate_stream(X, Y, 'nnfs', param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names).show()
