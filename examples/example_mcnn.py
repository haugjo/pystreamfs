from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import mcnn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a dataset
data = pd.read_csv('../datasets/drift.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = mcnn.run_mcnn

# Define parameters
param = dict()
param['num_features'] = 10  # number of features to return
param['batch_size'] = 100  # batch size
param['r'] = 20  # shifting window range for computation of stability

# Original parameters from paper
param['max_n'] = 100  # maximum number of saved instances per cluster
param['e_threshold'] = 3  # error threshold for splitting of a cluster
# Additional parameters
param['max_out_of_var_bound'] = 0.3  # percentage of variables that can at most be outside of variance boundary before new cluster is created
param['p_diff_threshold'] = 50  # threshold of perc. diff. for split/death rate when drift is assumed (_detect_drift())

# Define a ML model and a performance metric
model = SVC()
metric = accuracy_score

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, 'Micro Cluster Nearest Neighbor (MCNN)', type(model).__name__, metric.__name__, param).show()
