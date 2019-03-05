<img src="https://raw.githubusercontent.com/haugjo/pystreamfs/master/logo.png" width="450" height="70"/>

[![PyPI version](https://badge.fury.io/py/pystreamfs.svg)](https://badge.fury.io/py/pystreamfs)

<h1>!!! DEVELOPMENT RELEASE !!!</h1>

*pystreamfs* is an Open-Source Python package that allows for quick and simple comparison of feature selection algorithms on a simulated data and feature stream.

The user can simulate data/feature streams with varying batch size on any dataset provided as a numpy.ndarray. 
*pystreamfs* applies a specified feature selection algorithm to every batch and computes performance metrics for the
selected feature set at every time *t*. *pystreamfs* can also be used to plot the performance metrics.

*pystreamfs* comes with 5 built-in feature selection algorithms for data streams. Additionally, you can find 3 datasets ready for download on [Github](https://github.com/haugjo/pystreamfs). 
*pystreamfs* has a modular structure and is thus easily expandable (see Section 2.5 for more information).

**License:** MIT License<br>
**Upcoming changes:**
* ability to generate artificial data streams
* ability to test multiple feature selection algorithms at once

## 1 Getting started
### 1.1 Prerequesites
The following Python modules need to be installed (older versions than indicated might also work):
* python >= 3.7.1
* numpy >= 1.15.4
* psutil >= 5.4.7
* matplotlib >= 2.2.3
* scikit-learn >= 0.20.1
* ... any modules required by the feature selection algorithm 

### 1.2 How to get *pystreamfs*
Using pip: ``pip install pystreamfs``<br>
**OR** Download and unpack the .tar.gz file in ``/dist``. Navigate to the unpacked folder and execute
``python setup.py install``.
 
## 2 The Package  
### 2.1 Files
The main module is ``/pystreamfs/pystreamfs.py``. Feature selection algorithms are stored in ``/algorithms``.
 
### 2.2 Main module: ``pystreamfs.py``
``pystreamfs.py`` provides the following functions:
* ``X, Y = prepare_data(data, target, shuffle)``
    * **Description**: Prepare the data set for the simulation of a data stream: randomly sort the rows of a data matrix and extract the target variable ``Y`` and the features ``X``
    * **Input**:
        * ``data``: numpy.ndarray, data set
        * ``target``: int, index of the target variable
        * ``shuffle``: bool, if ``True`` sort samples randomly
    * **Output**:
        * ``X``: numpy.ndarray, features
        * ``Y``: numpy.ndarray, target variable
* ``stats = simulate_stream(X, Y, fs_algorithm, model, param)``
    * **Description**: Iterate over all datapoints in the dataset to simulate a data stream. 
    Perform given feature selection algorithm and return performance statistics.
    * **Input**:
        * ``X``: numpy.ndarray, this is the ``X`` returned by ``prepare_data()``
        * ``Y``: numpy.ndarray, this is the ``Y`` returned by ``prepare_data()``
        * ``fs_algorithm``: function, feature selection algorithm
        * ``ml_model``: object, the machine learning model to use for the computation of the accuracy score 
        (remark on KNN: number of neighbours has to be greater or equal to batch size)
        * ``param``: dict, includes:
            * ``num_features``: integer, the number of features you want returned
            * ``batch_size``: integer, the number of instances processed in one iteration
            * ... additional algorithm specific parameters
            * **optional**: ``feature_stream``: dict, key/value-pairs. Add an entry for every *t* where you want a change of available features:
                * key: time *t*
                * value: list, feature indices available at time *t*<br>
                **!!!Note: feature stream needs further testing and development!!!** 
    * **Output**:
        * ``stats``: dict
            * ``features``: list of lists, set of selected features for every batch
            * ``time_avg``: float, average computation time for one execution of the feature selection
            * ``time_measures``: list, time measures for every batch
            * ``memory_avg``: float, average memory usage after one execution of the feature selection, uses ``psutil.Process(os.getpid()).memory_full_info().uss``
            * ``memory_measures``: list, memory measures for every batch
            * ``acc_avg``: float, average accuracy for classification with the selected feature set
            * ``acc_measures``: list, accuracy measures for every batch
            * ``fscr_avg``: float, average feature selection change rate (fscr) per time window. 
            The fscr is the percentage of selected features that changes in *t* with respect to *t-1* (fscr=0 if all selected features remain the same, fscr=1 if all selected features change)
            * ``fscr_measures`` list, fscr measures for every batch
* ``plt = plot_stats(stats, ftr_names, param, fs_name, model_name):``
    * **Description**: Plot the statistics for time, memory, fscr and selected features over all time windows.
    * **Input**:
        * ``stats``: dict (see ``stats`` of ``simulate_stream()``)
        * ``ftr_names``: numpy.ndarray, contains all feature names
        * ``param``: dict, parameters
        * ``fs_name``: string, name of feature selection algorithm
        * ``model_name``: string, name of machine learning model
    * **Output**:
        * ``plt``: pyplot object: statistic plots

### 2.3 Built-in feature selection algorithms
* Online Feature Selection (OFS) based on the Perceptron algorithm by Wang et al. (2013) - [link to paper](https://ieeexplore.ieee.org/abstract/document/6522405)
* Unsupervised Feature Selection on Data Streams (FSDS) using matrix sketching by Huang et al. (2015) - [link to paper](https://dl.acm.org/citation.cfm?id=2806521)
* Feature Selection based on Micro Cluster Nearest Neighbors by Hamoodi et al. (2018) - [link to paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705118304039)
* Extremal Feature Selection based on a Modified Balanced Winnow classifier by Carvalho et al. (2006) - [link to paper](https://dl.acm.org/citation.cfm?id=1150466)
* CancelOut Feature Selection based on a Neural Network by Vadim Borisov ([Github](https://github.com/unnir/CancelOut))
    
### 2.4 Downloadable datasets
All datasets are cleaned and normalized. The target variable of all datasets is moved to the first column.
* German Credit Score ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))
* Binary version of Human Activity Recognition ([link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)).
    * The original HAR dataset has a multivariate target. For its binary version we defined the class "WALKING" as our positive class (label=1) and all other classes as the negative (non-walking) class. 
    We combined the 1722 samples of the original "WALKING" class with a random sample of 3000 instances from all other classes.
* Usenet ([link](http://www.liaad.up.pt/kdus/products/datasets-for-concept-drift))

### 2.5 How to add a feature selection algorithm
If you want to use *pystreamfs* to test your own feature selection algorithm, you have to encapsulate your algorithm in a function
with the following format:
```python
def your_fs_algorithm(X, Y, w, param):
    """Your feature selection algorithm
    
    :param numpy.nparray X: current data batch
    :param numpy.nparray Y: labels of current batch
    :param numpy.nparray w: feature weights
    :param dict param: any parameters the algorithm requires
    :return: w (updated feature weights), param
    :rtype numpy.ndarray, dict
    """

    ...do feature selection...
    
    return w, param
```
Afterwards you can import and test your feature selection algorithm in the same way as for any built-in algorithm (see the example).

## 3. Example
```python
from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import ofs
from sklearn.neighbors import KNeighborsClassifier

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
param['num_features'] = 5  # number of features to return
param['batch_size'] = 50  # batch size

# Define ML model
model = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)

# Data stream simulation
stats = pystreamfs.simulate_stream(X, Y, fs_algorithm, model, param)

# Plot statistics
pystreamfs.plot_stats(stats, feature_names, param, 'Online feature selection (OFS)', 'K Nearest Neighbor').show()
```
