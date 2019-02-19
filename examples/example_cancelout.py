from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import cancelout
from sklearn.svm import SVC

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Load a FS algorithm
fs_algorithm = cancelout.run_cancelout

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size

# Define a ML model
model = SVC()

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, param, 'CancelOut', 'Support Vector Classifier').show()
