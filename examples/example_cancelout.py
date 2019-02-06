from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import cancelout

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Load a FS algorithm
algorithm = cancelout.run_cancelout

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size
param['algorithm'] = 'svm'  # classification algorithm: here SVM

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, algorithm, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names).show()
