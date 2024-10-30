from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from Experiments.experiments import configJSON2obj
from Utils.vocabulary import (
    DATALOADER,
    DEVICE,
    INDEX,
    LABELS,
    LOSS,
    MODEL,
    PRED_POS,
    PREDICTED,
    REPRESENTATION,
    TEST,
    TRAIN,
    TRAIN_LABELS,
    VAL,
)


def one_infer(input, exp_config, exp_path, model_name="last_model.pt"):
    """
    Infer input through the model in the exp_path under the name model_name.
    exp_config is only used for the device attribute.
    """
    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Path {exp_path} doesn't exist"
    # Load model (not the one from the config file)
    device = exp_config[DEVICE] if torch.cuda.is_available() else torch.device("cpu")
    with open(exp_path/model_name, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.to(device)
    model.eval()

    input = input.to(device)
    pred = model(input) #Shape B, C, H, W
    pred_final = [pred[i].detach().cpu() for i in range(len(pred))]


    return pred_final

def inference(exp_config):
    """
    Return predictions (and representations if the model has a latent representation) for a model and a dataloader.


    Arguments:
    ----------
    exp_config: dict of Object. A configuration dictionary with instantiated objects. Keys DEVICE, MODEL, TRAIN_LABELS and DATALOADER are used.

    Returns:
    ---------
    predictions:
        mode
            label_i
                Labels: array
                Predicted: array
            index: array
    
    representations:
        mode
            representations: array
            labels
                label_key: array
            index: array
    """
    device = exp_config[DEVICE]
    model = exp_config[MODEL].to(device)
    has_latent = hasattr(model, 'latent_representation')

    predictions = {TRAIN: {}, VAL: {}}

    if has_latent:
        print(f"[inference] model {model.__class__.__name__} has a latent representation.")
        representations = {
            TRAIN: {REPRESENTATION: [], LABELS: {}, INDEX: []},
            VAL: {REPRESENTATION: [], LABELS: {}, INDEX: []},
        }

    for i in range(len(exp_config[TRAIN_LABELS][LOSS])):
        predictions[TRAIN][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
        predictions[TRAIN][INDEX] = []
        predictions[VAL][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
        predictions[VAL][INDEX] = []
        if has_latent:
            representations[TRAIN][LABELS][exp_config[TRAIN_LABELS][LABELS][i]] = []
            representations[VAL][LABELS][exp_config[TRAIN_LABELS][LABELS][i]] = []

    # Infer
    model.eval()
    for i, batch in tqdm(enumerate(exp_config[DATALOADER][TRAIN])) :
        x, labels, info = batch
        x = x.to(device)

        pred = model(x)

        # Store predictions
        for i, label_key in enumerate(set(exp_config[TRAIN_LABELS][LABELS]).intersection(set(labels.keys()))):
            pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i]
            
            # Store prediction
            predictions[TRAIN][label_key][PREDICTED].append(pred[pred_pos].detach().cpu().numpy())
            predictions[TRAIN][label_key][LABELS].append(labels[label_key])
        predictions[TRAIN][INDEX].append(info[INDEX])
        
        # Store last representation
        if has_latent:
            representations[TRAIN][REPRESENTATION].append(model.latent_representation.detach().cpu().numpy())
            for key in set(representations[TRAIN][LABELS].keys()).intersection(set(labels.keys())):
                representations[TRAIN][LABELS][key].append(labels[key])
            representations[TRAIN][INDEX].append(info[INDEX])

    # Validation
    for i, batch in tqdm(enumerate(exp_config[DATALOADER][TEST])) :
        x, labels, info = batch
        x = x.to(device)
        pred = model(x)
        
        for i, label_key in enumerate(exp_config[TRAIN_LABELS][LABELS]):
            pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i]

            # Store prediction
            predictions[VAL][label_key][PREDICTED].append(pred[pred_pos].detach().cpu().numpy())
            predictions[VAL][label_key][LABELS].append(labels[label_key])
        predictions[VAL][INDEX].append(info[INDEX])
        
        # Store last representation
        if has_latent:
            representations[VAL][REPRESENTATION].append(model.latent_representation.detach().cpu().numpy())
            for key in representations[VAL][LABELS].keys():
                representations[VAL][LABELS][key].append(labels[key])
            representations[VAL][INDEX].append(info[INDEX])

    # Concatenating predictions
    for mode, mode_dict in predictions.items():
        for label_key, label_val in mode_dict.items():
            if label_key == INDEX:
                if len(label_val) > 0:
                    predictions[mode][label_key] = np.concatenate(label_val, 0)
            else:
                for key, val in label_val.items():
                    if len(val) > 0:
                        predictions[mode][label_key][key] = np.concatenate(val, 0)

    # Concatenating representations
    if has_latent:
        for mode, mode_dict in representations.items():
            for label_key, label_val in mode_dict.items():
                if label_key == LABELS:
                    for key, val in label_val.items():
                        if len(val) :
                            representations[mode][label_key][key] = np.concatenate(val, 0)
                else:
                    if len(label_val) > 0:
                        representations[mode][label_key] = np.concatenate(label_val, 0)
        
        return predictions, representations
    else:
        return predictions



###############################################################################
###############################################################################

if __name__ == "__main__":
    """
    cf. Scripts/infer.py
    """