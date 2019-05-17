from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import ofs
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Load a dataset
data = pd.read_csv('../datasets/har.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = ofs.run_ofs

# Define parameters
param = dict()
param['num_features'] = 10  # number of features to return
param['batch_size'] = 100  # batch size
param['r'] = 25  # shifting window range for computation of stability

# Define a ML model and a performance metric
model = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=5, criterion='gini')
metric = roc_auc_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'Online feature selection (OFS)', type(model).__name__, metric.__name__, param, 0.8).show()
