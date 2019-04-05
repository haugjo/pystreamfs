from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import rfs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Normalize data
X = MinMaxScaler().fit_transform(X)

# Load a FS algorithm
fs_algorithm = rfs.run_rfs

# Define parameters
param = dict()
param['batch_size'] = 30
param['num_features'] = 5
param['epochs'] = 15  # iterations over current batch during one execution of rfs
param['mini_batch_size'] = 20  # must be smaller than batch_size
param['lr_my'] = 0.01  # learning rate for mean
param['lr_sigma'] = 0.01  # learning rate for standard deviation

# Define a ML model and a performance metric
model = SVC()
metric = accuracy_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'Robust Feature Selection', type(model).__name__, metric.__name__, param, 0.8).show()
