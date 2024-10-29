# Weakly-supervised semantic space structuring: cardiac cycle position for cerebral emboli visualization using contrastive learning

## Introduction
This repository presents the implementation of the experimnets of the following proceeding:
[](). A notebook to analyse category continuity metric with synthetic data is also proposed.

## Overview
### Category continuity
Category continuity is a metric proposed in the paper, which is a slight variation of [Pauwel et al. paper](https://www.sciencedirect.com/science/article/pii/S1077314299907634). It is implemented in the file [src/utils/metrics.py](src/utils/metrics.py). A notebook to present some properties of this metric is proposed at [notebooks/categoryContinuity_syntheticData.ipynb](notebooks/categoryContinuity_syntheticData.ipynb). This notebook contains saved plot, but is also mad eto be interactive: don't hesitate to download it and play qith the parameters thanks to [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/) module !

### Semantic structuring using contrastive learning
Our paper presents the integration of a semantic information (position in cardiac cycle-) of a medical event (cerebral emboli detection on a transcranial Doppler signal) thanks to contrastive learning. Triplet learning, contrastive learning with a generic loss have been implemented and compared. Auto-encoder was used as a baselin and a joint training was also performed.

The associate code thus provide:
* architecutre implementations,
* classes for dataset loading and pre-processing,
* training and evaluation scripts,
* configuration file to reproduce experiments
* notebooks for data plotting, including reproducing the plot of the paper.

Data is not provioded as it is private data from patients and no public dataset was used. 


