from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import nn_ubfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = nn_ubfs.run_nn_ubfs

# Define parameters
param = dict()
param['batch_size'] = 50
param['num_features'] = 5
param['r'] = 25  # shifting window range for computation of stability
param['epochs'] = 5  # iterations over current batch during one execution of ubfs
param['mini_batch_size'] = 10  # must be smaller than batch_size
param['lr_mu'] = 10  # learning rate for mean
param['lr_sigma'] = 10  # learning rate for standard deviation

param['lr_w'] = 10  # learning rate for weights
param['lr_lambda'] = 10  # learning rate for lambda

param['L'] = 10  # samples for monte carlo simulation
param['h'] = 50  # nodes of hidden layer

# Define a feature stream
feature_stream = dict()
feature_stream[0] = range(0, 15)
feature_stream[10] = range(0, 16)
feature_stream[11] = range(0, 17)  # TODO allow for multiple new features
param['feature_stream'] = feature_stream

# Define a ML model and a performance metric
model = SGDClassifier(loss='log')  # RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = roc_auc_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'NN-UBFS', type(model).__name__, metric.__name__, param, 0.8).show()
