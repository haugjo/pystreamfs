# streamfs
This module provides different feature selection (FS) algorithms for data streams.
The module allows you to simulate a data stream based on a provided data set. 
It further allows you to apply different feature selection algorithms to that very data stream.
Finally, the module provides you with a set of performance statistics of the selected feature selection algorithm.

## How to get the module
You can find the current distribution of ``streamfs`` for download in ``/dist``. 
Download and unpack the .zip (Windows) or .tar.gz (Linux) file. Navigate to the unpacked folder and execute
``python setup.py install`` to install the module.

## Requirements
You need a Python 3.x environment and the following packages to use the ``streamfs`` module:
 <br>``numpy, psutil``
 
## Functions
``streamfs`` provides the following functions:
* ``X, Y = prepare_data(data, target, shuffle)``
    * **Description**: Prepare the data set for the simulation of a data stream: randomly sort the rows of a the data matrix and extract the target variable ``Y`` and the features ``X``
    * **Input**:
        * ``data``: numpy array, this is the data set
        * ``target``: integer, index of the target variable
        * ``shuffle``: boolean, if ``True`` sort samples randomly
    * **Output**:
        * ``X``: numpy array, contains the features
        * ``Y``: numpy array, contains the target variable
* ``w, stats = simulate_stream(X, Y, algorithm, num_features)``
    * **Description**: Iterate over all datapoints in the dataset to simulate a data stream. 
    Perform given feature selection algorithm and return an array containing the weights for each (selected) feature as well as a set of performance statistics
    * **Input**:
        * ``X``: numpy array, this is the ``X`` returned by ``prepare_data()``
        * ``Y``: numpy array, this is the ``Y`` returned by ``prepare_data()``
        * ``algorithm``: string, this is the FS algorithm you want to apply
        * ``num_features``: integer, this is the number of features that should be returned
    * **Output**:
        * ``ftr_weights``: numpy array, contains the weights of the (selected) features
        * ``stats``: python dictionary
            * ``time_avg``: float, average computation time for one execution of the FS algorithm
            * ``time_measures``: list, individual computation time for each iteration (length = length of ``X``)
            * ``memory_avg``: float, average memory usage (relative to the total physical memory) for one execution of the FS algorithm
            * ``memory_start``: float, memory usage (relative to the total physical memory) before start of the data stream
            * ``memory_measures``: list, individual memory usage for each iteration (length = length of ``X``)

## FS algorithms
* Online Feature Selection (OFS) by Wang ([paper](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3277&context=sis_research))
    * ``algorithm = 'ofs'``
    
## Data Sets
* German Credit Score ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))

## Example
```python
import streamfs
import numpy as np

# Load german credit score dataset
credit_data = np.genfromtxt('./data/german_credit_score.csv', delimiter=';')

# Define the number of features that you want to return
num_features = 5

# Extract features and target variable
X, Y = streamfs.prepare_data(credit_data, 0)

# Simulate feature selection on a data stream (for the given data, FS algorithm and number of features)
w, stats = streamfs.simulate_stream(X, Y, 'ofs', num_features)

# Print resulting feature weights
print('Feature weights:\n', w)

# Print the average memory usage for one iteration of the FS algorithm
# -> this uses psutil.Process(pid).memory_percent()
print('Average memory usage: {}% (of total physical memory)'.format(stats['memory_avg'] * 100))

# Print average computation time in milliseconds for one iteration of the FS algorithm
print('Average computation time: {}ms'.format(stats['time_avg']))
```
             
 