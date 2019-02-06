from pystreamfs import pystreamfs_new
import numpy as np
import pandas as pd
from pystreamfs.algorithms import nnfs

# Load a dataset
data = pd.read_csv('../datasets/usenet.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Load a FS algorithm
algorithm = nnfs.run_nnfs

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 500  # batch size
param['algorithm'] = 'svm'  # classification algorithm: here SVM

# Extract features and target variable
X, Y = pystreamfs_new.prepare_data(data, 0, False)

# Data stream simulation
w, stats = pystreamfs_new.simulate_stream(X, Y, algorithm, param)

# Plot statistics
pystreamfs_new.plot_stats(stats, feature_names).show()
