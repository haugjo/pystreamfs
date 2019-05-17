from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import fsds
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/moa.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = fsds.run_fsds

# Define parameters
param = dict()
param['batch_size'] = 100  # batch size for one iteration, must be at least the same size than the no. of clusters!!
param['num_features'] = 3
param['r'] = 20  # shifting window range for computation of stability

param['B'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = 2  # no. of singular values (can be equal to no. of clusters/classes -> here 2 for binary class.)
param['m'] = data.shape[1]-1  # no. of original features

# Define a ML model and a performance metric
model = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = roc_auc_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'Feature Selection on Data Streams (FSDS)', type(model).__name__, metric.__name__, param, 0.8).show()
