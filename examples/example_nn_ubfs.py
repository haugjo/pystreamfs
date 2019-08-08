from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import nn_ubfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/kdd.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = nn_ubfs.run_nn_ubfs

# Define parameters
param = dict()
param['batch_size'] = 100
param['num_features'] = 15
param['r'] = 25  # shifting window range for computation of stability
param['epochs'] = 5  # iterations over current batch during one execution of ubfs
param['mini_batch_size'] = 30  # must be smaller than batch_size
param['lr_mu'] = 1  # learning rate for mean
param['lr_sigma'] = 0.1  # learning rate for standard deviation
param['lr_model'] = 0.01  # learning rate for SGD in neural net

param['lr_w'] = 10  # learning rate for weights
param['lr_lambda'] = 100  # learning rate for lambda Todo: think of temporarily increasing the learning rate after detection of new feature/class
param['init_lambda'] = 1

param['L'] = 10  # samples for monte carlo simulation
param['h'] = 50  # nodes of hidden layer

# Define a feature stream
feature_stream = dict()
feature_stream[0] = range(0, 10)
feature_stream[25] = range(0, 15)
feature_stream[50] = range(0, 20)
# param['feature_stream'] = feature_stream

# Define a ML model and a performance metric
model = Perceptron()  # RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = accuracy_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'NN-UBFS', type(model).__name__, metric.__name__, param, 0.8).show()
