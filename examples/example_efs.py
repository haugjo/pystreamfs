from pystreamfs import stream_simulator
import numpy as np
import pandas as pd
from pystreamfs.fs_algorithms import efs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a dataset
data = pd.read_csv('../datasets/moa.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = stream_simulator.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = efs.run_efs

# Define parameters
param = dict()
param['batch_size'] = 100  # batch size for one iteration, must be at least the same size than the no. of clusters!!
param['num_features'] = 10
param['r'] = 25  # shifting window range for computation of stability

# all parameters are set according to Carvalho et al.
param['u'] = np.ones(X[0].shape) * 2  # initial positive model with weights 2
param['v'] = np.ones(X[0].shape)  # initial negative model with weights 1
param['alpha'] = 1.5  # promotion parameter
param['beta'] = 0.5  # demotion parameter
param['threshold'] = 1  # threshold parameter
param['M'] = 1  # margin

# Define a ML model and a performance metric
model = SVC()
metric = accuracy_score

# Data stream simulation
stats = stream_simulator.simulate_stream(X, Y, fs_algorithm, model, metric, param)

# Plot statistics
stream_simulator.plot_stats(stats, feature_names, 'Extremal Feature Selection', type(model).__name__, metric.__name__, param, 0.8).show()
