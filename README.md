# Weakly-supervised semantic space structuring: cardiac cycle position for cerebral emboli visualization using contrastive learning

## Introduction
This repository presents the implementation of the experiments of a proceeding under review. A notebook to analyse category continuity metric with synthetic data is also proposed.

## Overview
### Category continuity
Category continuity is a metric proposed in the paper, which is a slight variation of [Pauwel et al. paper](https://www.sciencedirect.com/science/article/pii/S1077314299907634)'s NN-norm. It is implemented in the file [src/utils/metrics.py](src/utils/metrics.py). A notebook to present some properties of this metric is proposed at [notebooks/categoryContinuity_syntheticData.ipynb](notebooks/categoryContinuity_syntheticData.ipynb). This notebook contains saved plot, but is also made to be interactive: don't hesitate to download it and play with the parameters thanks to [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/) module ! This notebook can also be used with only the notebook and [metrics methods](src/Utils/metrics.py), if you do not want to download the whole folder.

### Semantic structuring using contrastive learning
Our paper presents the integration of a semantic information (position in cardiac cycle) of a medical event (cerebral emboli detection on a transcranial Doppler signal) thanks to contrastive learning. Triplet learning and contrastive learning with a generic loss have been implemented and compared. Auto-encoder was used as a baseline and a joint training of the autoencoder with a contrastive framework was also performed.

The associate code thus provide:
* architecture implementations,
* classes for dataset loading and pre-processing,
* training and evaluation scripts,
* configuration file to reproduce experiments
* trained model,
* notebooks for data plotting, including reproducing the plots of the paper.

**Data and trained models are not provided** as it is private data from patients and no public dataset was used.

## Configuration
The file [requirements](requirements.txt) enables to install the desired libraries, for example using pip:
```
pip install -r requirements.txt
```


Scripts are then available to train and evaluate data. A toy dataset has been created for demonstration purposes.

**1. Python path**

Add the path of your project to the python path:

```
export PYTHONPATH="${PYTHONPATH}:/path/to/project_name/src"
```

**2. Dataset splits**

(first time using a dataset) If there is no HDF5 file associated to your dataset (for example data/dataset_name/data.hdf5), create it using:

```
python src/DataManipulation/hdf5_data.py -p /path/to/data/dataset_name -g
```

Other arguments enable to choose some subjects to remove and some subjects for the test set. With no arguments, all subjects will be used for training.

**3. Configuration file**

Create or re-use a configuration file from [configs](/configs/) directory. Especially, the path to the HDF5 file that describes the dataset splits and paths is configured in this file. The file [config_file_detail](/src/config_file_detail.md) details the different configuration keys and available values.


**4. Experiment folder**

Create a folder for your experiment in the [experiments](/experiments/) folder.

**3. Training** 

Train your experiment 5 times (for example):

```
python src/Scripts/train_config.py -p /path/to/configs/exp_ConvAE_test.json -e /path/to/experiments/Exp_ConvAE_test -n 5
```


**4. Evaluation**

Evaluate the experiment results, using the evaluation part of the configuration file:

```
python src/Scripts/evaluate_exp.py -p /path/to/experiments/Exp_ConvAE_test -a
```

(-a to evaluate all the runs of the experiments)

OR using an evaluation config file:

```
python src/Scripts/evaluate_exp.py -p /path/to/experiments/Exp_ConvAE_test -a -m /path/to/configs/metric_classcontinuity_k10.json
```

**5. Plot results**

An example of results exploitation is given in a [notebook](/notebooks/Compare_expreriments.ipynb). The script [plot_losses](src/Results/plot_losses.py) also enables to plot the losses for one or several experiments and runs. The "detailed" argument is used to display only the total loss or the total loss and the different terms.


## Data
To use this code, your data has to follow the following BIDS structure (words under bracket are chosen by the user, * for optional files):

```
|_ data
    |_ [dataset_name]
        |_ [data].hdf5 (generated from the code)
        |_ contrastive_file.json (generated from the code)
        |_ participants.csv (subjects metadata)
        |_ sub-0
            |_ ses-1
                |_ sub-0_ses-1_run-1
                    |_ sub-0_ses-1_run-1.json (vignette metadata)
                    |_ sub-0_ses-1_run-1.PNG (vignette)
                    *|_ sub-0_ses-1_run-1.WAV (vignette audio)
        |_ sub-1
        |_ ...
```

## Executables
The most important functions are implemented in dedicated scripts in the [Scripts](src/Scripts) folder:
* [train_config](/src/Scripts/train_config.py) to train an experiment from a JSON configuration file (see [config_file_detail](/src/config_file_detail.md)) and a folder path to dump experiments results.
* [evaluate_exp](/src/Scripts/evaluate_exp) to evaluate an experiment from its JSON configuration file by default or a given configuration file ("evaluation" key of [config_file_detail](/src/config_file_detail.md)) and the experiment folder.
* [infer](/src/Scripts/infer) to infer a model from a dataset, given a dataset configuration, a model and a configuration for the other keys.

Some files implement methods and also contain a `main` function that can be called with arguments:
* [DataManipulation](src/DataManipulation/)
    * [Contrastive file](src/DataManipulation/contrastive_file.py) to generate or load the contrastive file associated to a dataset, which lists the groups of positives from the dataset
    * [HDF5 data](src/DataManipulation/hdf5_data.py) to generate or load an HDF5 file describing a dataset
    * [projection](src/DataManipulation/projection.py) to test projection functions
    * [representations](src/DataManipulation/representations.py) to test representations functions

* [Results](src/Results/)
    * [Plot losses](src/Results/plot_losses.py) to plot the losses of one or many runs, in a detailed way or not.

* [Utils](src/Utils/)
    * [Projection file](src/Utils/projection_file.py) to format samples 2D projection in a table with metadata (primarily used for a visualization UI).

