from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import random_benchmark
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/moa.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = random_benchmark.run_random_benchmark

# Define parameters
param = dict()
param['batch_size'] = 100
param['num_features'] = 25
param['r'] = 25  # shifting window range for computation of stability
param['epochs'] = 5  # iterations over current batch during one execution of ubfs
param['mini_batch_size'] = 30  # must be smaller than batch_size
param['lr_mu'] = 10  # learning rate for mean
param['lr_sigma'] = 10  # learning rate for standard deviation

# Define a ML model and a performance metric
model = Perceptron()  # RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = roc_auc_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'Random Benchmark', type(model).__name__, metric.__name__, param, 0.8).show()
