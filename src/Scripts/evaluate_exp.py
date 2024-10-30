import argparse
import json
import pickle as pkl
from pathlib import Path

from Experiments.evaluate import evaluate
from Experiments.experiments import configJSON2obj
from Experiments.inference import inference
from Utils.vocabulary import EVALUATION, REPRESENTATION

###############################################################################
###############################################################################

if __name__ == "__main__":
    """
    Script to evaluate an experiment given its configuration and inference results
    """
    parser = argparse.ArgumentParser(description="Evaluation script help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-a",  "--all",  action="store_true", required=False, default=False,
                        help="If all the runs in the parent folder have to be considered.")
    parser.add_argument("-f",  "--exp_folders", type=str, nargs="+", required=False,
                        help="Path to the experiments folder from the root folder, if all is false.")
    parser.add_argument("-m",  "--metric_config", type=str, required=False, default=None,
                        help="Path to the evaluation configuration (json)")

    # Interpreting arguments
    args = parser.parse_args()
    parent_folder = Path(args.parent_folder)
    assert parent_folder.exists()
    if args.all :
        folders = [parent_folder/exp_folder for exp_folder in parent_folder.glob('*') if (parent_folder/exp_folder).is_dir()]
    else:
        assert args.exp_folders, "Please enter run(s) folder(s) if '--all' is set to false"
        exp_folders = args.exp_folders
        folders = [parent_folder/exp_folder for exp_folder in exp_folders]
        
    # Getting external metric config if provided
    if args.metric_config is not None:
        metric_path = Path(args.metric_config)
        assert metric_path.exists(), f"[evaluate_exp_some_metrics] Metric config {metric_path} does not exist."
        print(f"[evaluate] Using metric list from {metric_path.parent.stem}/{metric_path.stem}")
        with metric_path.open("r") as f:
            metric_config = json.load(f)
    else:
        metric_config = None

    for run_path in folders:
        print(f"=====Evaluating experiment {run_path.parent.stem}-{run_path.stem}======")
        # Loading config
        with open(run_path/"config.json", "r") as f:
            json_config = json.load(f)
        obj_config = configJSON2obj(json_config)
        
        if metric_config is not None:
            # Replacing metric
            obj_config[EVALUATION] = metric_config

        print("|||||||||||||||||||", obj_config["device"])
        print(f"[evaluate_exp] Train dataset size: {len(obj_config['dataset']['train'])}")
        print(f"[evaluate_exp] Test dataset size: {len(obj_config['dataset']['test'])}")

        # Load predictions
        repr_path = run_path/"representations.pkl"
        pred_path = run_path/"predictions.pkl"
        if pred_path.exists():
            with pred_path.open("rb") as f:
                predictions = pkl.load(f)
        else:
            if repr_path.exists():
                predictions, representations = inference(obj_config, run_path)
            else:
                predictions = inference(obj_config, run_path)

        # Load representations
        representations = None
        # Representations path may not exist for model without latent space
        if repr_path.exists():
            with repr_path.open("rb") as f:
                representations = pkl.load(f) # Overwrite representations if computed in the previous "else"
            metrics = evaluate(obj_config, run_path, predictions, representations)
        else: 
            metrics = evaluate(obj_config, run_path, predictions)


