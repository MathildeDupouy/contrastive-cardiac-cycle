from pathlib import Path
import json
import torch
from tqdm import tqdm
import pickle as pkl
from DataManipulation.hdf5_data import ID, SAMPLE
from Experiments.experiments import get_dataloaders
from utils.vocabulary import (
    TRAIN, TEST, LABELS,
    DEVICE, INFO, MODEL,
    ID, SAMPLE,
    REPRESENTATION
)

def get_representation(model, dataloader, device = "cpu"):
    """
    Get the latent representation of a given model and dataset by inferring the model

    Arguments:
    ----------
        model: torch.Model
        The model to infer from. This model needs to have an attribute "latent_representation"

        dataloader: torch.utils.data.DataLoader
        A dataloader containing the data to infer from.

        device: str, optional
        The device to perform the inference
    
    Returns:
    --------
        torch.Tensor: representation of the data in a single tensor (batch are concatenated along the batch dimension)

        dict: Dictionary with label_key keys and torch.Tensor as values

        list: list of subject ids corresponding to the batched representations

        list: list of sample numbers corresponding to the batched representations
    """
    print("======Computing representations======")

    model = model.to(device)
    model.eval()

    first_batch = next(iter(dataloader))
    x, label, info = first_batch
    x = x.to(device)
    model(x) # Inference

    representations = model.latent_representation
    labels = label
    ids = info[ID]
    samples = info[SAMPLE]
    for batch in tqdm(dataloader):
        x, labels_i, info = batch
        x = x.to(device)
        model(x) # Inference
        representation = model.latent_representation

        # Gathering all information
        representations = torch.cat((representations, representation), 0)
        for key in labels.keys():
           labels[key] = torch.cat((labels[key], labels_i[key]), 0)
        ids = ids + info[ID]
        samples = samples + info[SAMPLE]

    print("======Representations computed======")

    return representations, labels, ids, samples

def load_representation(representation_path):
    """
    Load representation informations from a representation file.

    Arguments:
    ----------
        representation_path: str or Path
        A path to the representation hdf5

    Returns:
    --------
        dict: a dictionary with
            mode
                representation
                labels
                id
                sample number
    """
    representation_path = Path(representation_path)
    assert representation_path.is_file()

    with open(representation_path, 'rb') as f:
        representation_dict = pkl.load(f)
    return representation_dict

def get_exp_representation(exp_path, model = None, dataloaders = None, device = None, representation_name = "representations.pkl"):
    """
    Get the representations in the latent space corresponding to an experiment folder from the experiment config and last model.
    It infers the models and saves it if it doesn't already exists or load it from the existing file otherwise.

    Arguments:
    ----------
        exp_path: str or Path
        Path to the experiment folder (with config.json and last_model.pt)

        model: torch model, optional
        A model can be given to avoid loading the model. 
        Warning: if representations are load from file, no guarantee that this is the same model.

        dataloaders: dict of DataLoader by mode, optional
        Dataloaders can be given to avoid loading them again. 
        Warning: if representations are load from file, no guarantee that this is the same dataloaders.

        device: torch device, optional
        Device can be given to avoid loading it from config file. 
        Warning: if representations are load from file, no guarantee that this is the same device.

    Returns:
    --------
        dict: The representation dictionary
            mode
                REPRESENTATION: Tensor with all the representations in a batch
                LABELS: dict of tensors of labels
                ID: list of ids
                SAMPLE: list of sample numbers

    """
    exp_path = Path(exp_path)

    representation_path = exp_path/representation_name

    if not representation_path.exists():
        if dataloaders is None or device is None:
            # Getting the configuration
            config_path = exp_path/"config.json"
            assert config_path.exists()
            with config_path.open('r') as f:
                json_config = json.load(f)
            if dataloaders is None:
                dataloaders, _, _ = get_dataloaders(json_config)
            if device is None:
                device = json_config[DEVICE]

        if model is None:
            # Getting the model
            model_path = exp_path/"last_model.pt"
            assert model_path.exists()
            model = torch.load(model_path)
            model.to(device)

        # Generating the representation hdf5 file
        representation_dict = {}
        for mode in [TRAIN, TEST]:

            dl = dataloaders[mode]
            representations, labels, ids, samples = get_representation(model, dl, device)

            representation_dict[mode] = {
                REPRESENTATION: representations,
                LABELS: labels,
                ID: ids,
                SAMPLE: samples
            }

        if dataloaders is None or device is None:
            # Saving representation size in the json config if loaded
            if INFO not in json_config[MODEL].keys():
                json_config[MODEL][INFO] = {REPRESENTATION: representations.shape[1:]} # Remove batch size
            else:
                json_config[MODEL][INFO][REPRESENTATION] = representations.shape[1:]
            with config_path.open('w') as f:
                json_config = json.dump(json_config, f, indent = 4)

        # Create representation file
        with open(representation_path, 'wb') as f:
            pkl.dump(representation_dict, f)
    else:
        #Reading the representation file already existing
        print(f"[representations] Representation path {representation_path.stem} for experience {'/'.join(representation_path.parts[-3:-1])} already exists, loading it.")
        representation_dict = load_representation(representation_path)

    return representation_dict

###############################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment projection help")
    parser.add_argument("-p",  "--exp_path", type=str, nargs="+", required=True,
                        help="Experiment paths to the experiment folders to be projected")

    args = parser.parse_args()
    exp_paths = args.exp_path

    for exp_path in exp_paths:
        get_exp_representation(exp_path)


