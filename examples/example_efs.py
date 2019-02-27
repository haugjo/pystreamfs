from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import efs
from sklearn.svm import SVC

# Load a dataset
data = pd.read_csv('../datasets/usenet.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = efs.run_efs

# Define parameters
param = dict()
param['batch_size'] = 50  # batch size for one iteration, must be at least the same size than the no. of clusters!!
param['num_features'] = 5

# all parameters are set according to Carvalho et al.
param['u'] = np.ones(X[0].shape) * 2  # initial positive model with weights 2
param['v'] = np.ones(X[0].shape)  # initial negative model with weights 1
param['alpha'] = 1.5  # promotion parameter
param['beta'] = 0.5  # demotion parameter
param['threshold'] = 1  # threshold parameter
param['M'] = 1  # margin

# Define a ML model
model = SVC()

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, param, 'Extremal Feature Selection (EFS)', 'Support Vector Classifier').show()
