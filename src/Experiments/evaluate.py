import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, silhouette_score
from tqdm import tqdm

from DataManipulation.projection import get_exp_projection
from DataManipulation.representations import get_exp_representation
from Experiments.experiments import configJSON2obj
from Utils.metrics import category_continuity
from Utils.tools import update_saved_dict
from Utils.vocabulary import (
    ALL,
    ARGS,
    DATASET,
    DEVICE,
    EVALUATION,
    HARD_CLASS,
    HARD_POS,
    INDEX,
    LABELS,
    METRIC,
    PREDICTED,
    PROJECTION,
    REPRESENTATION,
    SOFT_CLASS,
    SOFT_POS,
    TEST,
    TRAIN,
    UNSUPERVISED,
    VAL,
)

ELEMENTWISE_METRICS = ["MSE"]
POINTCLOUD_METRICS = ["silhouette_class_latent", "silhouette_class_2d", "silhouette_pos_latent", "silhouette_pos_2d", "silhouette_detailed_latent", "silhouette_detailed_2d",
                      "categorycontinuity_class_latent", "categorycontinuity_class_2d", "categorycontinuity_pos_latent", "categorycontinuity_pos_2d", "categorycontinuity_detailed_latent", "categorycontinuity_detailed_2d"]

def evaluate(exp_config, exp_path, predictions, representations = None):
    """
    Evaluate an experiment based on the objects instantiated in exp_config and an experiment path.
    
    Arguments:
    ----------
        exp_config: dictionary
            Experiment configuration, generally loaded from a configuration file
        exp_path: str or Path
            Path to the folder to store the generated files
        predictions: dictionary
            The predictions associated to the dataset of the configuration
    Returns:
    --------
        dictionary of metrics
    """
    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Path {exp_path} doesn't exist"
    modes = {TRAIN: exp_config[DATASET][TRAIN]}
    if VAL in predictions.keys():
        modes[TEST] = exp_config[DATASET][TEST]
    metrics = {ALL: {}, TRAIN: {}, TEST: {}}
    for idx, metric in enumerate(exp_config[EVALUATION][ALL][METRIC]):
        print(f"======Evaluating metric {metric}======")
        args = exp_config[EVALUATION][ALL][ARGS][idx] if ARGS in exp_config[EVALUATION][ALL].keys() else None

        #--- MSE -----------#
        if metric.lower() == "mse":
            metric_fn = torch.nn.MSELoss()

            # Train
            true_train = predictions[TRAIN][UNSUPERVISED][LABELS]
            pred_train = predictions[TRAIN][UNSUPERVISED][PREDICTED]
            metrics[TRAIN][metric.lower()] = float(metric_fn(torch.Tensor(pred_train), torch.Tensor(true_train)))

            # Test
            if VAL in predictions.keys():
                true_test = predictions[VAL][UNSUPERVISED][LABELS]
                pred_test = predictions[VAL][UNSUPERVISED][PREDICTED]
                metrics[TEST][metric.lower()] = float(metric_fn(torch.Tensor(pred_test), torch.Tensor(true_test)))

                # All
                metrics[ALL][metric.lower()] = float(metric_fn(torch.Tensor(np.concatenate((pred_train, pred_test), 0)), torch.Tensor(np.concatenate((true_train, true_test), 0))))
            else:
                # All
                metrics[ALL][metric.lower()] = float(metric_fn(torch.Tensor(pred_train), torch.Tensor(true_train)))

        #--- POINTCLOUD METRICS -----------#
        elif metric.lower() in POINTCLOUD_METRICS:
            args, s_metrics = pointcloud_evaluation(metric, exp_path, exp_config, modes, predictions, representations, args)
            metric_name = metric.lower() + "_" + "_".join([ f"{k}{v}".replace("_", "-") for k, v in args.items()])
            metrics = {k: {**v, metric_name: s_metrics[k]} for k, v in metrics.items() if k in s_metrics.keys()}
        #--- MCC -----------#
        elif "mcc" in metric.lower():
            # Getting labels
            if "class" in metric.lower():
                label_key = HARD_CLASS
            elif "pos" in metric.lower():
                label_key = HARD_POS
            else:
                raise NotImplementedError(f"[evaluate] MCC metric '{metric}' not implemented (classification unknown).")

            pred_all = []
            labels_all = []
            for mode in modes.keys():
                mode2 = VAL if mode == TEST else TRAIN
                # Hard classification straight from the network
                if label_key in predictions[mode2].keys():
                    pred = predictions[mode2][label_key][PREDICTED]
                    labels = predictions[mode2][label_key][LABELS]
                else:
                    softlabel_key = SOFT_CLASS if label_key == HARD_CLASS else SOFT_POS
                    if softlabel_key in predictions[mode2].keys():
                        pred = torch.argmax(torch.Tensor(predictions[mode2][softlabel_key][PREDICTED]), dim = 1).numpy()
                        labels = torch.argmax(torch.Tensor(predictions[mode2][softlabel_key][LABELS]), dim = 1).numpy()
                    else:
                        print(f"No label keys {softlabel_key} ({label_key})")
                metrics[mode][metric.lower()] = matthews_corrcoef(labels, pred)

                pred_all.append(pred)
                labels_all.append(labels)
            metrics[ALL][metric.lower()] = matthews_corrcoef(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0))
        else:
            raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented.")


    # Saving metrics
    metric_path = exp_path/"metric.json"
    if metric_path.exists():
        update_saved_dict(metric_path, metrics)
    else:
        with metric_path.open("w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


def pointcloud_evaluation(metric, exp_path, exp_config, modes_datasets, predictions, representations = None, args = None):
    """
    Evaluate predictions with a given pointcloud metric based on the objects instantiated in exp_config and an experiment path.
    
    Arguments:
    ----------
        metric: string
            Metric name, containing 'categorycontinuity' or 'silhouette', '2d' or 'latent', '_class', '_pos' or 'detailed'
        exp_config: dictionary
            Experiment configuration, generally loaded from a configuration file
        exp_path: str or Path
            Path to the folder to store the generated files
        predictions: dictionary
            The predictions associated to the dataset of the configuration
    Returns:
    --------
        dictionary of metrics
    """
    metrics = {}
    if "2d" in metric.lower():
        projection_dict = get_exp_projection(exp_path, device=exp_config[DEVICE])
    elif "latent" in metric.lower() and representations is None:
        representations = get_exp_representation(exp_path)
    else:
        raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented (no space specified).")

    X_all = []
    labels_all = []
    for mode, dataset in modes_datasets.items():
        mode2 = VAL if mode == TEST else TRAIN # TODO val and test
        # Getting data in the desired space
        if "latent" in metric.lower():
            # Flatten representations
            X = representations[mode2][REPRESENTATION].reshape(representations[mode2][REPRESENTATION].shape[0], -1)
        elif "2d" in metric.lower():
            X = projection_dict[mode2][PROJECTION]
        else:
            raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented (no space specified).")
        
        # Getting labels
        if "class" in metric.lower():
            label_key = HARD_CLASS
        elif "pos" in metric.lower():
            label_key = HARD_POS
        elif "detailed" in metric.lower():
            label_key = "detailed"
        else:
            raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented (no true label type specified).")
        if label_key == "detailed":
            labels = np.array([str(dataset[idx][1][HARD_CLASS]) + str(str(dataset[idx][1][HARD_POS])) for idx in predictions[mode2][INDEX]])
            # print('detailed silhouette', {key : np.count_nonzero(labels == key) for key in set(labels)})
            # print('detailed silhouette', len(np.unique(labels)), np.unique(labels))
        else:
            labels = np.array([dataset[idx][1][label_key] for idx in predictions[mode2][INDEX]])
        X_all.append(X)
        labels_all.append(labels)

        # Computing score
        if "silhouette" in metric.lower():
            if args is None:
                args = {'metric': 'euclidean'}
            metrics[mode] = float(silhouette_score(X, labels, **args))
        elif "categorycontinuity" in metric.lower():
            if args is None:
                args = {'k': 10, 'aggregation': 'total_mean', 'sigma': None}
            metrics[mode] = float(category_continuity(X, labels, **args)[0])
        else:
            raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented (wrong metric name).")

    if "silhouette" in metric.lower():
        if args is None:
            args = {'metric': 'euclidean'}
        metrics[ALL] = float(silhouette_score(np.concatenate(X_all, 0), np.concatenate(labels_all, 0), **args))
    elif "categorycontinuity" in metric.lower():
        if args is None:
            args = {'k': 10, 'aggregation': 'total_mean', 'sigma': None}
        metrics[ALL] = float(category_continuity(np.concatenate(X_all, 0), np.concatenate(labels_all, 0), **args)[0])
    else:
        raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented (wrong metric name).")

    return args, metrics


###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""
        cf. script in Scripts/evaluate_exp.py
          """)