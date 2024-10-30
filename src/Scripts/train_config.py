
"""
Run this script to launch an experiment from a configuration JSON file. 
See config_file_detail.md to know the different key and values necessary and available in a configuration file.
Arguments:
----------
    -e --experiments, int, multiple, required
    id of the experiments to run

    -n --number_run, int, optional (default 1)
    Number of training of each experiment
"""
# TODO default values of a configuration file
import argparse
import json
from pathlib import Path

from Experiments.experiments import (
    configJSON2obj,
    create_run_folder,
    get_model,
    get_optimizer,
)
from Experiments.train import train
from Utils.vocabulary import DEVICE, MODEL, OPTIMIZER

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments test help")
    parser.add_argument("-p",  "--config_path", type=str, nargs="+", required=True,
                        help="Path to the JSON configuration file")
    parser.add_argument("-e",  "--exp_path", type=str, nargs="+", required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-n",  "--number_run", type=int, required=False, default=1,
                        help="Number of runs for each experiments")

    args = parser.parse_args()

    assert len(args.exp_path) == len(args.config_path), f"You did not give the same number of configuration and experiment paths ({len(args.config_path)} configs, {len(args.exp_path)} paths)."

    for config_path, exp_path in zip(args.config_path, args.exp_path):
        config_path = Path(config_path)
        exp_path = Path(exp_path)
        assert config_path.exists(), f"Config path {config_path} does not exist."
        assert exp_path.exists(), f"Config path {exp_path} does not exist."

        with Path(config_path).open("r") as f:
            json_config = json.load(f)

        # Translating to a configuration dict with instantiated objects
        obj_config = configJSON2obj(json_config)
        print(f"[Experiments] Asking for the device {json_config[DEVICE]}, final device {obj_config[DEVICE]}")

        for i in range(args.number_run):
            print(f"=====Training experiment {exp_path.stem}, {i+1}th run======")
            run_path = create_run_folder(exp_path)
            # Save configuration
            with (Path(run_path)/"config.json").open("w") as f:
                json.dump(json_config, f, indent = 4)

            # Train
            obj_config[MODEL] = get_model(json_config)
            obj_config[OPTIMIZER] = get_optimizer(json_config, obj_config[MODEL])
            train(obj_config, run_path)