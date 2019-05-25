from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import ubfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/usenet.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = ubfs.run_ubfs

# Define parameters
param = dict()
param['batch_size'] = 100
param['num_features'] = 10
param['r'] = 25  # shifting window range for computation of stability
param['epochs'] = 5  # iterations over current batch during one execution of ubfs
param['mini_batch_size'] = 30  # must be smaller than batch_size
param['lr_mu'] = 10**5  # learning rate for mean
param['lr_sigma'] = 10**2  # learning rate for standard deviation

# Parameters for concept drift detection
param['check_drift'] = True  # indicator whether to check drift or not
param['range'] = 2  # range of last t to check for drift

# Define a ML model and a performance metric
model = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = roc_auc_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'UBFS', type(model).__name__, metric.__name__, param, 0.8).show()
