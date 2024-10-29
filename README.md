# Weakly-supervised semantic space structuring: cardiac cycle position for cerebral emboli visualization using contrastive learning

## Introduction
This repository presents the implementation of the experiments of the following proceeding:
[](). A notebook to analyse category continuity metric with synthetic data is also proposed.

## Overview
### Category continuity
Category continuity is a metric proposed in the paper, which is a slight variation of [Pauwel et al. paper](https://www.sciencedirect.com/science/article/pii/S1077314299907634). It is implemented in the file [src/utils/metrics.py](src/utils/metrics.py). A notebook to present some properties of this metric is proposed at [notebooks/categoryContinuity_syntheticData.ipynb](notebooks/categoryContinuity_syntheticData.ipynb). This notebook contains saved plot, but is also mad eto be interactive: don't hesitate to download it and play with the parameters thanks to [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/) module ! Notebook can be used with only the notebook and metrics scripts, if you do not want to download the whole folder.

### Semantic structuring using contrastive learning
Our paper presents the integration of a semantic information (position in cardiac cycle-) of a medical event (cerebral emboli detection on a transcranial Doppler signal) thanks to contrastive learning. Triplet learning, contrastive learning with a generic loss have been implemented and compared. Auto-encoder was used as a baseline and a joint training was also performed.

The associate code thus provide:
* architecture implementations,
* classes for dataset loading and pre-processing,
* training and evaluation scripts,
* configuration file to reproduce experiments
* trained model,
* notebooks for data plotting, including reproducing the plots of the paper.

Data is not provided as it is private data from patients and no public dataset was used.

## Configuration

## Data
To use this code, your data has to follow the following BIDS structure (words under bracket are chosen by the user, * for optional files):

```
|_ data
    |_ [dataset_name]
        |_ sub-0
            |_ ses-1
                |_ sub-0_ses-1_run-1
                    |_ sub-0_ses-1_run-1.json
                    |_ sub-0_ses-1_run-1.PNG
                    *|_ sub-0_ses-1_run-1.WAV
        |_ sub-1
        |_ ...
        |_ [data].hdf5 (generated from the code)
        |_ contrastive_file.json (generated from the code)
```

## Use
The most important functions are implemented in dedicated scripts in the [Scripts](src/Scripts) folder:

Some folders implements method and also contains a main function that can be called with arguments:
* [DataManipulation](src/DataManipulation/)
    * [Contrastive file](src/DataManipulation/contrastive_file.py) to generate or load the contrastive file associated to a dataset, which lists the groups of positives from the dataset
    * [HDF5 data](src/DataManipulation/hdf5_data.py) to generate or load an HDF5 file describing a dataset
    * [projection](src/DataManipulation/projection.py) to test projection functions
    * [representations](src/DataManipulation/reppresentations.py) to test representations functions

