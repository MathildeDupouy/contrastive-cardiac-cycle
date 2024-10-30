import json
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import tqdm

from Utils.vocabulary import (
    CHECKPOINT_FREQ,
    CONTRASTIVE,
    DATALOADER,
    DEVICE,
    INDEX,
    LABELS,
    LOSS,
    MODEL,
    NB_EPOCHS,
    OPTIMIZER,
    PRED_POS,
    PREDICTED,
    REPRESENTATION,
    TEST,
    TOTAL,
    TRAIN,
    TRAIN_LABELS,
    UNSUPERVISED,
    VAL,
    WEIGHT,
)


def train(exp_config, run_path, last_model_name = "last_model.pt"):
    """
    Method to train a model on the training dataset, and evaluate the losses values on the validation dataset for a given number of epochs.
    It stores the files "[last_model_name].pt", "losses.json", predictions.pkl (and representations.pkl if relevant). 
    Predictions and representations are obtained with one inference of the last model.
    If the configuration file contains a CHECKPOINT_FREQ key, models are saved at a given frequency.
    Detailed return files:
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

    Loss_values: 
        mode
            loss_i: [...]
            total: [...]
            
    Arguments:
    ----------
    exp_config: dict of Object
    Configuration dictionary of instantiated objects
    
    run_path: string of Path
    Path to the run folder
    
    last_model_name: str
    name for the last model file
    """
    run_path = Path(run_path)
    device = exp_config[DEVICE]
    model = exp_config[MODEL].to(device)
    optimizer = exp_config[OPTIMIZER]

    predictions = {TRAIN: {}}
    loss_values = {TRAIN: {TOTAL:[]}}
    if hasattr(model, 'latent_representation'):
        representations = {
            TRAIN: {REPRESENTATION: [], INDEX: []},
        }
    if len(exp_config[DATALOADER][TEST]) > 0:
        predictions[VAL] = {}
        loss_values[VAL] = {TOTAL:[]}
        if hasattr(model, 'latent_representation'):
            representations[VAL] = {REPRESENTATION: [], INDEX: []}

    for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
        loss_values[TRAIN][loss_fn.__class__.__name__] = []
        predictions[TRAIN][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
        predictions[TRAIN][INDEX] = []
        if len(exp_config[DATALOADER][TEST]) > 0:
            loss_values[VAL][loss_fn.__class__.__name__] = []
            predictions[VAL][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
            predictions[VAL][INDEX] = []

    checkpoints = []
    if CHECKPOINT_FREQ in exp_config.keys():
        checkpoints = [i for i in range(exp_config[CHECKPOINT_FREQ] - 1, exp_config[NB_EPOCHS] - 1, exp_config[CHECKPOINT_FREQ])]
        print("[train] checkpoints epochs:", checkpoints)
    epoch_bar = tqdm.tqdm(range(exp_config[NB_EPOCHS] + 1)) # Last epoch without optimization, just to infer last model
    for epoch in epoch_bar:
        # Training
        model.train()
        mean_loss = {key:0 for key in loss_values[TRAIN].keys()}
        mean_loss[TOTAL] = 0
        for i, batch in enumerate(exp_config[DATALOADER][TRAIN]) :
            epoch_bar.set_description(f"Batch {i + 1}/{len(exp_config[DATALOADER][TRAIN])}")

            x, labels, info = batch

            optimizer.zero_grad()

            if type(x) in (tuple, list):
                for i in range(len(x)):
                    x[i] = x[i].to(device)
                pred = model(*x)
            else:
                x = x.to(device)
                pred = model(x)

            # Construct loss
            loss = 0
            for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                # Loss get, add and store value
                label_key = exp_config[TRAIN_LABELS][LABELS][i]
                pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i]
                weight = exp_config[TRAIN_LABELS][WEIGHT][i]
                if label_key == CONTRASTIVE:
                    assert type(pred_pos) in (tuple, list)
                    loss_inputs = ()
                    for pred_posi in pred_pos:
                        loss_inputs += (pred[pred_posi],)
                else:
                    target = labels[label_key].to(device).squeeze()
                    loss_inputs = (pred[pred_pos], target)
                loss_i = weight * loss_fn(*loss_inputs)
                loss = loss + loss_i
                
                # Store prediction
                if epoch == exp_config[NB_EPOCHS] or epoch in checkpoints:
                    if type(pred_pos) in (tuple, list):
                        preds = []
                        for pred_posi in pred_pos:
                            preds += [pred[pred_posi].detach().cpu()]
                        predictions[TRAIN][label_key][PREDICTED].append(preds)
                        predictions[TRAIN][label_key][LABELS].append(np.empty(1))
                    else:
                        predictions[TRAIN][label_key][PREDICTED].append(pred[pred_pos].detach().cpu().numpy())
                        predictions[TRAIN][label_key][LABELS].append(labels[label_key])
                # Store loss
                if epoch < exp_config[NB_EPOCHS]:
                    mean_loss[loss_fn.__class__.__name__] += loss_i.item()

            # Optimize (except inference epoch)
            if epoch < exp_config[NB_EPOCHS]:
                loss.backward()
                optimizer.step()

            # Store last prediction
            if epoch == exp_config[NB_EPOCHS] or epoch in checkpoints:
                predictions[TRAIN][INDEX].append(info[INDEX])
                
                # Store last representation
                if hasattr(model, 'latent_representation'):
                    representations[TRAIN][REPRESENTATION].append(model.latent_representation.detach().cpu().numpy())
                    representations[TRAIN][INDEX].append(info[INDEX])
            if epoch < exp_config[NB_EPOCHS] :
                # Store loss
                mean_loss[TOTAL] += loss.item()
    
        if epoch < exp_config[NB_EPOCHS]:
            for loss_key, loss_val in mean_loss.items():
                loss_values[TRAIN][loss_key].append(loss_val / len(exp_config[DATALOADER][TRAIN]))

        # Validation
        if len(exp_config[DATALOADER][TEST]) > 0:
            model.eval()
            mean_loss = {key:0 for key in loss_values[VAL].keys()}
            mean_loss[TOTAL] = 0
            for i, batch in enumerate(exp_config[DATALOADER][TEST]) :
                epoch_bar.set_description(f"Test {i + 1}/{len(exp_config[DATALOADER][TEST])}")
                x, labels, info = batch

                if type(x) in (tuple, list):
                    for i in range(len(x)):
                        x[i] = x[i].to(device)
                    with torch.no_grad():
                        pred = model(*x)
                else:
                    x = x.to(device)
                    with torch.no_grad():
                        pred = model(x)
                
                loss = 0
                for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                    label_key = exp_config[TRAIN_LABELS][LABELS][i]
                    weight = exp_config[TRAIN_LABELS][WEIGHT][i]
                    pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i]

                    if label_key == CONTRASTIVE:
                        assert type(pred_pos) in (tuple, list)
                        loss_inputs = ()
                        for pred_posi in pred_pos:
                            loss_inputs += (pred[pred_posi],)
                    else:
                        target = labels[label_key].to(device).squeeze()
                        loss_inputs = (pred[pred_pos], target)

                    loss_i = weight * loss_fn(*loss_inputs)
                    loss = loss + loss_i

                    # Store last prediction
                    if epoch == exp_config[NB_EPOCHS] or epoch in checkpoints:
                        if type(pred_pos) in (tuple, list):
                            preds = ()
                            for pred_posi in pred_pos:
                                preds += (pred[pred_posi].detach().cpu().numpy(),)
                            predictions[VAL][label_key][PREDICTED].append(preds)
                            predictions[VAL][label_key][LABELS].append(np.empty(1))
                        else:
                            predictions[VAL][label_key][PREDICTED].append(pred[pred_pos].detach().cpu().numpy())
                            predictions[VAL][label_key][LABELS].append(labels[label_key])

                    # Store loss
                    if epoch < exp_config[NB_EPOCHS]:
                        mean_loss[loss_fn.__class__.__name__] += loss_i.item()

                # Store last prediction
                if epoch == exp_config[NB_EPOCHS] or epoch in checkpoints:
                    predictions[VAL][INDEX].append(info[INDEX])
                    
                    # Store last representation
                    if hasattr(model, 'latent_representation'):
                        representations[VAL][REPRESENTATION].append(model.latent_representation.detach().cpu().numpy())
                        representations[VAL][INDEX].append(info[INDEX])
                if epoch < exp_config[NB_EPOCHS] :
                    # Store loss
                    mean_loss[TOTAL] += loss.item()

            if epoch < exp_config[NB_EPOCHS]:
                for loss_key, loss_val in mean_loss.items():
                    loss_values[VAL][loss_key].append(loss_val / len(exp_config[DATALOADER][TEST]))
                epoch_bar.set_postfix({"Loss": mean_loss})

        if epoch in checkpoints:
            # Save model
            model.eval()
            model.to("cpu")
            torch.save(model, run_path/f"model_ep{epoch}.pt")
            model.to(exp_config[DEVICE])

            # Saving predictions file
            for mode, mode_dict in predictions.items():
                for label_key, label_val in mode_dict.items():
                    if label_key == INDEX:
                        predictions[mode][label_key] = np.concatenate(label_val, 0)
                    else:
                        for key, val in label_val.items():
                            if type(val[0]) in (list, tuple):
                                pred_1_batch = val[0]
                                for p in range(1, len(val)):
                                    for i in range(len(val[0])):
                                        np.concatenate((pred_1_batch[i], val[p][i]), 0)
                            else:
                                predictions[mode][label_key][key] = np.concatenate(val, 0)
            with (run_path/f"predictions_ep{epoch}.pkl").open("wb") as f:
                pkl.dump(predictions, f)

            # Reinitialize predictions
            for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                loss_values[TRAIN][loss_fn.__class__.__name__] = []
                predictions[TRAIN][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
                predictions[TRAIN][INDEX] = []
                if len(exp_config[DATALOADER][TEST]) > 0:
                    loss_values[VAL][loss_fn.__class__.__name__] = []
                    predictions[VAL][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
                    predictions[VAL][INDEX] = []

            # Saving representations file
            if hasattr(model, 'latent_representation'):
                for mode, mode_dict in representations.items():
                    for label_key, label_val in mode_dict.items():
                        if label_key == LABELS:
                            for key, val in label_val.items():
                                representations[mode][label_key][key] = np.concatenate(val, 0)
                        else:
                            representations[mode][label_key] = np.concatenate(label_val, 0)
                with (run_path/f"representations_ep{epoch}.pkl").open("wb") as f:
                    pkl.dump(representations, f)
                # Reinitialize representations
                representations = {
                    TRAIN: {REPRESENTATION: [], INDEX: []},
                    VAL: {REPRESENTATION: [], INDEX: []},
                }

    assert run_path.exists()
    with (run_path/"losses.json").open("w") as f:
        json.dump(loss_values, f, indent = 4)

    # Save model
    model.to("cpu")
    torch.save(model, run_path/last_model_name)
    model.to(exp_config[DEVICE])

    # Saving predictions file
    for mode, mode_dict in predictions.items():
        for label_key, label_val in mode_dict.items():
            if label_key == INDEX:
                predictions[mode][label_key] = np.concatenate(label_val, 0)
            else:
                for key, val in label_val.items():
                    if type(val[0]) in (list, tuple):
                        pred_1_batch = val[0]
                        for p in range(1, len(val)):
                            for i in range(len(val[0])):
                                np.concatenate((pred_1_batch[i], val[p][i]), 0)
                    else:
                        predictions[mode][label_key][key] = np.concatenate(val, 0)
    with (run_path/"predictions.pkl").open("wb") as f:
        pkl.dump(predictions, f)

    # Saving representations file
    if hasattr(model, 'latent_representation'):
        for mode, mode_dict in representations.items():
            for label_key, label_val in mode_dict.items():
                if label_key == LABELS:
                    for key, val in label_val.items():
                        representations[mode][label_key][key] = np.concatenate(val, 0)
                else:
                    representations[mode][label_key] = np.concatenate(label_val, 0)
        with (run_path/"representations.pkl").open("wb") as f:
            pkl.dump(representations, f)
    
    return loss_values

###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""See corresponding scripts:
          - train_exp to train from an experiment number
          - train_config to train from a configuration file
          """)