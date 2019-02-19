<img src="logo.png" width="400" height="70"/>

*pystreamfs* is an Open-Source Python package that allows for quick and simple comparison of feature selection algorithms on a simulated data stream.

The user can simulate data streams with varying batch size on any dataset provided as a numpy.ndarray. 
*pystreamfs* applies a specified feature selection algorithm to every batch and computes performance metrics for the
selected feature set at every time *t*. *pystreamfs* can also be used to plot the performance metrics.

The package currently includes 3 datasets and 4 feature selection algorithms built in. 
*pystreamfs* has a modular structure and is thus easily expandable.

**Version:** 0.1.0<br>
**License:** MIT License<br>
**Upcoming changes:**
* additional built in datasets, feature selection algorithms and classifiers
* ability to simulate feature streams
* ability to generate artificial data streams

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
**OR** Download and unpack the .zip (Windows) or .tar.gz (Linux) file in ``/dist``. Navigate to the unpacked folder and execute
``python setup.py install``.
 
## 2 The Package  
### 2.1 Files
The main module is ``/pystreamfs/pystreamfs.py``. Feature selection algorithms are stored in ``/algorithms``. 
Datasets are stored in ``/datasets``. Examples can be found in ``/examples``.
 
### 2.2 Main module: ``pystreamfs.py``
``pystreamfs.py`` provides the following functions:
* ``X, Y = prepare_data(data, target, shuffle)``
    * **Description**: Prepare the data set for the simulation of a data stream: randomly sort the rows of a the data matrix and extract the target variable ``Y`` and the features ``X``
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
        * ``X``: numpy array, this is the ``X`` returned by ``prepare_data()``
        * ``Y``: numpy array, this is the ``Y`` returned by ``prepare_data()``
        * ``fs_algorithm``: function, feature selection algorithm
        * ``ml_model``: object, the machine learning model to use for the computation of the accuracy score 
        (remark on KNN: number of neighbours has to be greater or equal to batch size)
        * ``param``: python dict(), includes:
            * ``num_features``: integer, the number of features you want returned
            * ``batch_size``: integer, number of instances processed in one iteration
            * ... additional algorithm specific parameters
    * **Output**:
        * ``stats``: python dictionary
            * ``features``: set of selected features for every batch
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
        * ``stats``: python dictionary (see ``stats`` of ``simulate_stream()``)
        * ``ftr_names``: numpy array, contains all feature names
        * ``param``: python dict(), parameters
        * ``fs_name``: string, name of FS algorithm
        * ``model_name``: string, name of ML model
    * **Output**:
        * ``plt``: pyplot object: statistic plots

### 2.3 Built-in feature selection algorithms
* Online Feature Selection (OFS) by Wang et al. ([paper](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3277&context=sis_research))
* Unsupervised Feature Selection on Data Streams (FSDS) by Huang et al.([paper](http://www.shivakasiviswanathan.com/CIKM15.pdf))
* Feature Selection based on Micro Cluster Nearest Neighbors by Hamoodi et al. ([paper](https://www.researchgate.net/profile/Mahmood_Shakir2/publication/326949948_Real-Time_Feature_Selection_Technique_with_Concept_Drift_Detection_using_Adaptive_Micro-Clusters_for_Data_Stream_Mining/links/5b89149e4585151fd13e1b1a/Real-Time-Feature-Selection-Technique-with-Concept-Drift-Detection-using-Adaptive-Micro-Clusters-for-Data-Stream-Mining.pdf))
* CancelOut Feature Selection based on a Neural Network by Vadim Borisov (*more information will be included*)
    
### 2.4 Built-in datasets
All datasets are cleaned and normalized. The target variable of all datasets is moved to the first column.
* German Credit Score ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))
* Binary version of Human Activity Recognition ([link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)).
    * The original HAR dataset has a multivariate target. For its binary version we defined the class "WALKING" as our positive class (label=1) and all other classes as the negative (non-walking) class. 
    We combined the 1722 samples of the original "WALKING" class with a random sample of 3000 instances from all other classes.
* Usenet ([link](http://www.liaad.up.pt/kdus/products/datasets-for-concept-drift))


## 3. Example
```python
from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from pystreamfs.algorithms import ofs
from sklearn.neighbors import KNeighborsClassifier

# Load a dataset
data = pd.read_csv('../datasets/credit.csv')
feature_names = np.array(data.drop('target', 1).columns)
data = np.array(data)

# Extract features and target variable
X, Y = pystreamfs.prepare_data(data, 0, False)
Y[Y == 0] = -1  # change 0 to -1, required by ofs

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