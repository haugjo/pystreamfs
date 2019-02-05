from pystreamfs import pystreamfs
import numpy as np
import pandas as pd

# Load a dataset
data = pd.read_csv('../datasets/usenet.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Define parameters
param = dict()
param['num_features'] = 10  # number of features to return
param['batch_size'] = 50  # batch size
param['algorithm'] = 'svm'  # classification algorithm: here SVM

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)
Y[Y == 0] = -1  # change 0 to -1, required by ofs

# Data stream simulation
w, stats = pystreamfs.simulate_stream(X, Y, 'ofs', param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names).show()
