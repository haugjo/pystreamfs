import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from pystreamfs.pipeline import Pipeline
from pystreamfs.feature_selector import FeatureSelector

# Load a dataset
data = pd.read_csv('../datasets/har_binary.csv')
feature_names = np.array(data.drop('target', 1).columns)  # Todo: move to stream_simulator
dataset = np.array(data)

# Select FS algorithm
fs_param = dict()
fs_param['batch_size'] = 100
fs_param['num_features'] = 100
fs_param['r'] = 25  # shifting window range for computation of stability
fs_param['epochs'] = 5  # iterations over current batch during one execution of ubfs
fs_param['mini_batch_size'] = 25  # must be smaller than batch_size
fs_param['lr_mu'] = 0.1  # learning rate for mean
fs_param['lr_sigma'] = 0.1  # learning rate for standard deviation
fs_param['init_sigma'] = 1

fs_param['lr_w'] = 0.1  # learning rate for weights
fs_param['lr_lambda'] = 0.1  # learning rate for lambda
fs_param['init_lambda'] = 1

# Parameters for concept drift detection
fs_param['check_drift'] = False  # indicator whether to check drift or not
fs_param['range'] = 2  # range of last t to check for drift
fs_param['drift_basis'] = 'mu'  # basis parameter on which we perform concept drift detection

fs_algorithm = FeatureSelector('iufes', fs_param)

# Generate Pipeline
param = dict()
param['live_visual'] = False  # Todo: implement live visualization

pipe = Pipeline(dataset, None, fs_algorithm, Perceptron(), accuracy_score, param)

# Start Pipeline
pipe.start()

# Plot results
pipe.plot()
