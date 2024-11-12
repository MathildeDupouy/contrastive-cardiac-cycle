# TRANSLATE THE CONFIG FILE
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from DataManipulation.hdf5_data import load_from_HDF5
from DataManipulation.HITS_2D_dataset import HITS_2D_Dataset
from DataManipulation.HITS_contrastive_dataset import HITS_contrastive_Dataset
from Models.ClassifConvAE import ClassifConvAE
from Models.ContrastiveNet import ContrastiveNet, ContrastiveNetAE
from Models.ConvolutionalAE import (
    ConvolutionalAE,
    ConvolutionalDecoder,
    ConvolutionalEncoder,
)
from Models.tools import weights_init
from Models.TripletNet import TripletNet, TripletNetAE
from Utils.ContrastiveLoss import ContrastiveLoss, ContrastiveLossCosSim, TripletLoss
from Utils.vocabulary import *


def get_dataloaders(json_config):
    """
    Get information from the cofig DATASET, TRAIN_LABELS, TEST_LABELS and EVALUATION keys to build 
    the corresponding dataloaders.

    Arguments:
    ----------
        json_config: dict
        configuration file
    
    Returns:
    --------
        dict: keys are TRAIN and TEST and their values are torch dataloaders
    """
    assert Path(json_config[DATASET][DATA_PATH]).exists(), f"[experiments] Dataset path {json_config[DATASET][DATA_PATH]} does not exists"
    train_batch_size = json_config[BATCH_SIZE] if type(json_config[BATCH_SIZE]) == int else json_config[BATCH_SIZE][0]
    test_batch_size = json_config[BATCH_SIZE] if type(json_config[BATCH_SIZE]) == int else json_config[BATCH_SIZE][1]

    # Load data
    data = load_from_HDF5(json_config[DATASET][DATA_PATH])
    available_datasets = list(data.keys()) # In theory several datasets possible but in practice only one
    train_data = data[available_datasets[0]][TRAIN]
    test_data = data[available_datasets[0]][TEST]

    # Create datasets and dataloaders
    dataloaders = {}
    if "hits" in json_config[DATASET][DATASET_TYPE].lower():
        # Dataloader prepared for training and evaluation
        train_labels = list(set(json_config[TRAIN_LABELS][LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][TRAIN][LABELS]))
        test_labels = list(set(json_config[TEST_LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][TEST][LABELS]))

        if json_config[DATASET][DATASET_TYPE].lower() == "hits_2d":
            train_dataset = HITS_2D_Dataset(data=train_data, label_keys=train_labels, **json_config[DATASET][ARGS])
            test_dataset = HITS_2D_Dataset(data=test_data, label_keys=test_labels, **json_config[DATASET][ARGS])
        elif json_config[DATASET][DATASET_TYPE].lower() == "hits_contrastive":
            # Get contrastive dictionary
            contrastive_file_path = Path(json_config[DATASET][CONTRASTIVE_PATH])
            assert contrastive_file_path.exists(), f"[HITS contrastive dataset] The contrastive file {contrastive_file_path} does not exist."
            with contrastive_file_path.open("r") as f:
                contrastive_groups = json.load(f)
            train_dataset = HITS_contrastive_Dataset(data=train_data, contrastive_groups=contrastive_groups[TRAIN], label_keys=train_labels, **json_config[DATASET][ARGS])
            test_dataset = HITS_contrastive_Dataset(data=test_data, contrastive_groups=contrastive_groups[TEST], label_keys=test_labels, **json_config[DATASET][ARGS])
        else :
            raise NotImplementedError(f"[Experiments] Dataset config {json_config[DATASET][DATASET_TYPE]} is not implemented.")

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=False))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
        dataloaders[TRAIN] = train_loader
        dataloaders[TEST] = test_loader

        json_config[DATASET][INFO] = {
            "duration": train_dataset.duration,
            "width": train_dataset.width,
            "height": train_dataset.height,
            "train size": len(train_dataset),
            "test size": len(test_dataset),
            }
        # Dataloader prepared for training and evaluation
        train_labels = list(set(json_config[TRAIN_LABELS][LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][TRAIN][LABELS]))
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=False))
        dataloaders[TRAIN] = train_loader

    else:
        raise NotImplementedError(f"[Experiments] Dataset config {json_config[DATASET][DATASET_TYPE]} is not implemented.")

    return dataloaders, train_dataset, test_dataset

def get_model(json_config):
    """
    Get a model from the configuration dict, and initialize it.
    """
    if json_config[MODEL][NAME].lower() == "convolutionalae":
        model = ConvolutionalAE(**json_config[MODEL][ARGS])
    elif json_config[MODEL][NAME].lower() == "convencoder":
        model = ConvolutionalEncoder(**json_config[MODEL][ARGS])
    elif json_config[MODEL][NAME].lower() == "convdecoder":
        model = ConvolutionalDecoder(**json_config[MODEL][ARGS])
    elif json_config[MODEL][NAME].lower() == "classifconvae":
        # Get the iput dimension from the created dataset (cf. get_dataloaders)
        ch, height, width = 3, json_config[DATASET][INFO]["height"], json_config[DATASET][INFO]["width"]
        # "args" need to contain at least n_classes
        model = ClassifConvAE(input_shape=(ch, height, width), **json_config[MODEL][ARGS])
    elif json_config[MODEL][NAME].lower() == "triplet":
        # "args" need to contain at least MODEL for the embedding net
        embedding_net = get_model(json_config[MODEL][ARGS])
        model = TripletNet(embedding_net)
    elif json_config[MODEL][NAME].lower() == "contrastive":
        # "args" need to contain at least MODEL for the embedding net
        embedding_net = get_model(json_config[MODEL][ARGS])
        model = ContrastiveNet(embedding_net)
    elif json_config[MODEL][NAME].lower() == "tripletae":
        # "args" need to contain at least "encoder" and "decoder" for the embedding net
        encoder = get_model(json_config[MODEL][ARGS]["encoder"])
        decoder = get_model(json_config[MODEL][ARGS]["decoder"])
        model = TripletNetAE(encoder, decoder)
    elif json_config[MODEL][NAME].lower() == "contrastiveae":
        # "args" need to contain at least "encoder" and "decoder" for the embedding net
        encoder = get_model(json_config[MODEL][ARGS]["encoder"])
        decoder = get_model(json_config[MODEL][ARGS]["decoder"])
        model = ContrastiveNetAE(encoder, decoder)
    else:
        raise NotImplementedError(f"[Experiments] Model config {json_config[MODEL][NAME]} is not implemented.")
    model.apply(weights_init)
    return model

def get_optimizer(json_config, model):
    """
    Instantiate an optimizer from the configuration dict and a given model. 
    """
    if json_config[OPTIMIZER][NAME] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **json_config[OPTIMIZER][ARGS])
    else:
        raise NotImplementedError(f"[Experiments] optimizer config {json_config[OPTIMIZER][NAME]} is not defined.")
    return optimizer

def configJSON2obj(json_config):
    """
    Translate the JSON config dictionary to a dictionary with instantiated dataloaders, model, losses,
    except metrics.

    Arguments:
    ----------
        json_config: (dict)
        An experiment configuration file

    Returns:
    --------
        A configuration dict with instantiated objects.
    """
    obj_config = {}

    # Generating the configuration dict with objects
    ## GENERAL
    obj_config[NAME] = json_config[NAME]
    if "cuda" in json_config[DEVICE]:
        obj_config[DEVICE] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        obj_config[DEVICE] = torch.device("cpu")

    ## CHECKPOINTS
    if CHECKPOINT_FREQ in json_config.keys():
        obj_config[CHECKPOINT_FREQ] = json_config[CHECKPOINT_FREQ]
    ## NB_EPOCHS
    obj_config[NB_EPOCHS] = json_config[NB_EPOCHS]

    ## BATCH SIZE
    if isinstance(json_config[BATCH_SIZE], list) or isinstance(json_config[BATCH_SIZE], tuple):
        obj_config[BATCH_SIZE] = {
            TRAIN: json_config[BATCH_SIZE][0],
            TEST:  json_config[BATCH_SIZE][1],
        }
    elif isinstance(json_config[BATCH_SIZE], float):
        obj_config[BATCH_SIZE] = {
            TRAIN: json_config[BATCH_SIZE],
            TEST: json_config[BATCH_SIZE]
        }
    else:
        raise NotImplementedError(f"[Experiments] batch size config type {type(json_config[BATCH_SIZE])} is not handled.")

    ## DATASETS (external function)
    obj_config[DATASET] = {}
    obj_config[DATALOADER], obj_config[DATASET][TRAIN], obj_config[DATASET][TEST] = get_dataloaders(json_config)

    ## MODEL
    obj_config[MODEL] = get_model(json_config)

    ## OPTIMIZER
    obj_config[OPTIMIZER] = get_optimizer(json_config, obj_config[MODEL])

    ## LOSSES
    obj_config[TRAIN_LABELS] = {}
    obj_config[TRAIN_LABELS][LABELS] = json_config[TRAIN_LABELS][LABELS]
    obj_config[TRAIN_LABELS][WEIGHT] = json_config[TRAIN_LABELS][WEIGHT]
    obj_config[TRAIN_LABELS][PRED_POS] = json_config[TRAIN_LABELS][PRED_POS]
    obj_config[TRAIN_LABELS][LOSS] = []
    for loss, args in zip(json_config[TRAIN_LABELS][LOSS],json_config[TRAIN_LABELS][ARGS]):
        if loss.lower() == "mse":
            obj_config[TRAIN_LABELS][LOSS].append(torch.nn.MSELoss(**args))
        elif loss.lower() == "ce":
            obj_config[TRAIN_LABELS][LOSS].append(torch.nn.CrossEntropyLoss(**args))
        elif loss.lower() == "triplet":
            obj_config[TRAIN_LABELS][LOSS].append(TripletLoss(**args))
        elif loss.lower() == "contrastive":
            obj_config[TRAIN_LABELS][LOSS].append(ContrastiveLoss(**args))
        elif loss.lower() == "contrastivecossim":
            obj_config[TRAIN_LABELS][LOSS].append(ContrastiveLossCosSim(**args))
        else:
            raise NotImplementedError(f"[Experiments] Loss config {loss} is not implemented.")
        

    ## Evaluation
    obj_config[EVALUATION] = json_config[EVALUATION]

    return obj_config

def create_run_folder(exp_path, run_name = None):
    """
    Create a folder with the name run_name (or run-i) in the experiment folder exp_name.

    Arguments:
    ----------
    exp_name: str or Path
    path to the experiment folder, created if this folder does not exist

    run_name: str
    Name for the run

    Returns:
    ----------
    str:Path to rhe run folder
    """

    exp_path = Path(exp_path)
    if not exp_path.exists():
        os.mkdir(exp_path)

    i = 0
    if run_name is None:
        run_name = f"run-{i}"
        while (exp_path/run_name).exists():
            run_name = f"run-{i}"
            i += 1

    output_folder = exp_path/run_name
    os.mkdir(output_folder)

    return output_folder


###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""See corresponding scripts:
          - train_exp to train from an experiment number
          - train_config to train from a json configuration file
          """)