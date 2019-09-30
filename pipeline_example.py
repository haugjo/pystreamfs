import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from pystreamfs.pipeline import Pipeline
from pystreamfs.feature_selector import FeatureSelector

# Parameters
param = dict()

# Load a dataset
data = pd.read_csv('../datasets/har_binary.csv')
feature_names = np.array(data.drop('target', 1).columns)  # Todo: move to stream_simulator
dataset = np.array(data)
param['label_idx'] = 0
param['shuffle_data'] = False

# Select FS algorithm
fs_prop = dict()  # FS Algorithm properties
fs_prop['epochs'] = 5  # iterations over current batch during one execution of iufes
fs_prop['mini_batch_size'] = 25  # must be smaller than batch_size
fs_prop['lr_mu'] = 0.1  # learning rate for mean
fs_prop['lr_sigma'] = 0.1  # learning rate for standard deviation
fs_prop['init_sigma'] = 1

fs_prop['lr_w'] = 0.1  # learning rate for weights
fs_prop['lr_lambda'] = 0.1  # learning rate for lambda
fs_prop['init_lambda'] = 1

# Parameters for concept drift detection
fs_prop['check_drift'] = False  # indicator whether to check drift or not
fs_prop['range'] = 2  # range of last t to check for drift
fs_prop['drift_basis'] = 'mu'  # basis parameter on which we perform concept drift detection

fs_algorithm = FeatureSelector('iufes', fs_prop)

# Generate Pipeline
param['live_visual'] = False  # Todo: implement live visualization
param['batch_size'] = 100
param['num_features'] = 100
param['r'] = 25  # shifting window range for computation of stability

pipe = Pipeline(dataset, None, fs_algorithm, Perceptron(), accuracy_score, param)

# Start Pipeline
pipe.start()

# Plot results
pipe.plot()