"""
Run this script to infer a model from a dataset.
Arguments:
----------
    -p --parent_folder, str, required
    path to the parent folder of the run

    -o --output_path, str, required
    path to the output folder for predictions
    
    -m --model, str, required
    path ot the model path

    -dc --dataset_config, str, optional (default None)
    Path to the JSON file with the configuration from the dataset
    Exemple :
    {
        "dataset type": "HITS_2D",
        "data path": "/path/to/data/dataset_name/data.hdf5",
        "initial_duration": 1500,
        "duration": 400,
        "multiple": 16,
        "args": {
            "remove_pos_U": false,
            "remove_class_U": false
        }
    }

"""

import argparse
from pathlib import Path
import json
import pickle as pkl
import os

import torch

from Experiments.inference import inference
from Experiments.experiments import configJSON2obj
from Utils.vocabulary import (
    DEVICE,
    DATASET,
    MODEL,
)

if __name__ == "__main__":
    # Example : python src/Scripts/infer.py -dc /path/to/data/dataset_name/dataset_config.json
    #                                       -p /path/to/experiments/experiment_name/run-0
    #                                       -m /path/to/experiments/experiment_name/run-0/last_model.pt
    #                                       -o /path/to/experiments/inferences/temporal_sub_400ms_a1_b0-01_run-0
    parser = argparse.ArgumentParser(description="Inference script help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-o",  "--output_path", type=str, required=True,
                        help="Path to the folder where predictions will be stored")
    parser.add_argument("-m",  "--model", type=str, required=True,
                        help="Path to the model to infer from")
    parser.add_argument("-dc",  "--dataset_config", type=str, required=False, default=None,
                        help="Path to the JSON file with the configuration from the dataset")
    

    args = parser.parse_args()
    exp_folder = Path(args.parent_folder)
    model_path = Path(args.model)
    assert exp_folder.exists(), f"{exp_folder} does not exist."
    assert model_path.exists(), f"{model_path} does not exist."

    with (exp_folder/'config.json').open("r") as f:
        json_config = json.load(f)

    # Replacing dataset arguments
    if args.dataset_config is not None :
        with open(args.dataset_config, "r") as f:
            dataset_config = json.load(f)
        json_config[DATASET] = dataset_config


    # Translating to a configuration dict with instantiated objects
    obj_config = configJSON2obj(json_config)
    print(f"[Experiments] Asking for the device {json_config[DEVICE]}, final device {obj_config[DEVICE]}")
    
    ## Replacing model
    with model_path.open("r") as f:
        model = torch.load(f)
    obj_config[MODEL] = model

    predictions, representations = inference(obj_config)

    output_path = Path(args.output_path)
    os.mkdir(output_path)
    with (output_path/"predictions.pkl").open("wb") as f:
        pkl.dump(predictions, f)
    with (output_path/"representations.pkl").open("wb") as f:
        pkl.dump(representations, f)
    with (output_path/"config.json").open("w") as f:
        json.dump(json_config, f, indent = 4)

    