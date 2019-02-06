# PyStreamFS
PyStreamFS allows for quick and simple comparison of feature selection algorithms on a simulated data stream.

The user can simulate data streams with varying batch size on any dataset provided as a numpy array. 
PyStreamFS applies a specified feature selection algorithm to every batch and computes performance metrics for the
selected feature set at every time t. PyStreamFS can also be used to plot the performance metrics.

The package currently includes 3 datasets, 4 feature selection algorithms and 3 classifiers for computation of the accuracy scores.

**Version:** 0.0.1<br>
**Upcoming changes:**
* additional dataset, feature selection algorithms and classifiers
* ability to simulate feature streams
* ability to generate artificial data streams

## 1 Getting started
### 1.1 How to get the module
You can find the current distribution of ``pystreamfs`` for download in ``/dist``. 
Download and unpack the .zip (Windows) or .tar.gz (Linux) file. Navigate to the unpacked folder and execute
``python setup.py install`` to install the module.

### 1.2 Prerequesites
* python >= 3.7.1
* numpy >= 1.15.4
* pandas >= 0.23.4
* psutil >= 5.4.7
* matplotlib >= 2.2.3
* scikit-learn >= 0.20.2
 
## 2 The Package  
### 2.1 Files
The main module is ``pystreamfs/pystreamfs.py``. Feature selection algorithms are stored in ``algorithms``. 
Datasets are stored in ``datasets``.
 
### 2.2 Functions of pystreamfs.py
``pystreamfs`` provides the following functions:
* ``X, Y = prepare_data(data, target, shuffle)``
    * **Description**: Prepare the data set for the simulation of a data stream: randomly sort the rows of a the data matrix and extract the target variable ``Y`` and the features ``X``
    * **Input**:
        * ``data``: numpy.ndarray, data set
        * ``target``: int, index of the target variable
        * ``shuffle``: bool, if ``True`` sort samples randomly
    * **Output**:
        * ``X``: numpy.ndarray, features
        * ``Y``: numpy.ndarray, target variable
* ``stats = simulate_stream(X, Y, algorithm, param)``
    * **Description**: Iterate over all datapoints in the dataset to simulate a data stream. 
    Perform given feature selection algorithm and return performance statistics.
    * **Input**:
        * ``X``: numpy array, this is the ``X`` returned by ``prepare_data()``
        * ``Y``: numpy array, this is the ``Y`` returned by ``prepare_data()``
        * ``algorithm``: function, feature selection algorithm
        * ``param``: python dict(), includes:
            * ``num_features``: integer, the number of features you want returned
            * ``batch_size``: integer, number of instances processed in one iteration
            * ... additional algorithm specific parameters (check 2.3)
    * **Output**:
        * ``stats``: python dictionary
            * ``features``: set of selected features for every batch
            * ``time_avg``: float, average computation time for one execution of the feature selection
            * ``time_measures``: list, time measures for every batch
            * ``memory_avg``: float, average memory usage after one execution of the feature selection, uses ``psutil.Process(os.getpid()).memory_full_info().uss``
            * ``memory_measures``: list, memory measures for every batch
            * ``acc_avg``: float, average accuracy for classification with the selected feature set
            * ``acc_measures``: list, accuracy measures for every batch
            * ``fscr_avg``: HIER WEITER!!!!!!!!!!!
            * ``fscr_measures``
* ``plt = plot_stats(stats, ftr_names):``
    * **Description**: Plot the time and memory consumption as provided in stats. Also plot selected features over time.
    If the number of features is smaller or equal 30, this function plots a tic for each feature in the y-axis.
    * **Input**:
        * ``stats``: python dictionary (see ``stats`` of ``simulate_stream()``)
        * ``ftr_names``: numpy array, contains all feature names
    * **Output**:
        * ``plt``: pyplot object: contains a subplot for time/memory consumption and selected features over time

### 2.3 FS algorithms
* Online Feature Selection (OFS) by Wang et al. ([paper](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3277&context=sis_research))
    * ``algorithm = 'ofs'``
* Unsupervised Feature Selection on Data Streams (FSDS) by Huang et al.([paper](http://www.shivakasiviswanathan.com/CIKM15.pdf))
    * ``algorithm =  'fsds'``
    * Additional parameters included in ``param``:
        * ``b``: list, initial sketch matrix
        * ``ell``: integer, initial sketch size
        * ``k``: integer, number of singular values (usually equal to number of clusters)
    
### 2.4 Data Sets
* Cleaned version of German Credit Score dataset ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))
* Human Activity Recognition dataset ([link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones))

## 3. Adding new FS algorithms
If you want to add a new algorithm to the ``streamfs`` package follow these steps:
1. create a new python file in the ``algorithms`` folder: ``[acronym].py``.
2. check ``utils.py`` for existing utility functions (e.g. truncation of feature vector).
3. Implement your algorithm as function ``run_[acronym]()`` in ``[acronym].py``. 
The function has to return at least a vector of feature-weights + the computation time and memory consumption for one execution.
Comment your function with docstring. **In any case, check ``ofs.py`` or ``fsds.py`` for reference.**
4. Import ``algorithms/[acronym].py`` in ``streamfs.py``.
5. Add ``elif algorithm == [acronym]``-condition in ``simulate_stream()`` function.
Call ``run_[acronym]()`` from here (this executes your function for every "newly arrived" datapoint).
6. Add algorithm description to README (section 2.3)
7. Add example of algorithm usage to README (section 4)



## 4. Examples
### 4.1 OFS example
```python
from streamfs import streamfs
import numpy as np
import pandas as pd

# Load german credit score dataset
credit_data = pd.read_csv('../datasets/cleaned_german_credit_score.csv')
feature_names = np.array(credit_data.columns)
credit_data = np.array(credit_data)

# Define parameters
param = dict()
param['num_features'] = 5  # number of features to return
param['batch_size'] = 1  # batch size for one iteration of ofs

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 23, False)
Y[Y == 0] = -1  # change 0 to -1, required by ofs

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', param)

# Print resulting feature weights
print('Feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(feature_names[stats['features'][-1]]))

# Print params
print('Statistics for one execution of OFS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()
```

### 4.2 FSDS example
```python
from streamfs import streamfs
import numpy as np
import pandas as pd

# Load humane activity recognition
har_data = pd.read_csv('../datasets/human_activity_recognition.csv')
feature_names = np.array(har_data.columns)
har_data = np.array(har_data)

# Extract features and target variable
X, Y = streamfs.prepare_data(har_data, 562, False)
Y, clusters = pd.factorize(Y)
X = np.array(X, dtype='float')  # data has to be provided as float for internal SVD
Y = np.array(Y)

# Define initial parameters for FSDS
param = dict()
param['b'] = []  # initial sketch matrix
param['ell'] = 0  # initial sketch size
param['k'] = clusters.size  # no. of singular values (equal to no. of clusters)
param['batch_size'] = 1000  # batch size for one iteration
param['num_features'] = 10

# Simulate feature selection on a data stream (for the given data, FS algorithm and parameters)
w, stats = streamfs.simulate_stream(X, Y, 'fsds', param)

# Print resulting feature weights
print('Feature weights:\n', w[stats['features'][-1]])
print('Selected features: {}'.format(stats['features'][-1]))

# Print params
print('Statistics for one execution of FSDS with a batch size of {}:'.format(param['batch_size']))

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))

# Plot time and memory consumption
streamfs.plot_stats(stats, feature_names).show()
```