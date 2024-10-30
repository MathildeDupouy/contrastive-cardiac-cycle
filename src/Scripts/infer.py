"""
Run this script to infer a model from a dataset.
Arguments:
----------
    -p --parent_folder, str, required
    path to the parent folder of the run

    -o --output_path, str, required
    path to the output folder for predictions

    -dc --dataset_config, str, optional (default None)
    Path to the JSON file with the configuration from the dataset
    Exemple :
    {
        "dataset type": "HITS_2D",
        "data path": "/home/dupouy/Documents/stage_emboles/data/data_04_2024_1_5s/embStickers_1.5s_T0_rearranged_posMSec/data.hdf5",
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

from Experiments.inference import inference
from Experiments.experiments import configJSON2obj
from Utils.vocabulary import (
    DEVICE,
    DATASET,
)

if __name__ == "__main__":
    # Example : python src/Scripts/infer.py -dc /home/dupouy/Documents/stage_emboles/data/data_04_2024_1_5s/embStickers_1.5s_T0_rearranged_posMSec/dataset_config.json
    #                                       -p /home/dupouy/Documents/stage_emboles/code/learning_temporal_context/experiments/14-Exp_ConvAE_ClassifPos_a1_b0-01/run-0
    #                                       -o /home/dupouy/Documents/stage_emboles/code/learning_temporal_context/experiments/inferences/temporal_sub_400ms_a1_b0-01_run-0
    parser = argparse.ArgumentParser(description="Inference script help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-o",  "--output_path", type=str, required=True,
                        help="Path to the folder where predictions will be stored")
    parser.add_argument("-dc",  "--dataset_config", type=str, required=False, default=None,
                        help="Path to the JSON file with the configuration from the dataset")
    

    args = parser.parse_args()
    exp_folder = Path(args.parent_folder)
    assert exp_folder.exists(), f"{exp_folder} does not exist."

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

    predictions, representations = inference(obj_config)

    output_path = Path(args.output_path)
    os.mkdir(output_path)
    with (output_path/"predictions.pkl").open("wb") as f:
        pkl.dump(predictions, f)
    with (output_path/"representations.pkl").open("wb") as f:
        pkl.dump(representations, f)
    with (output_path/"config.json").open("w") as f:
        json.dump(json_config, f, indent = 4)

    