"""
Function and script to create a file with the link to png images sorted by subjects and position.
Script arguments :
-p : Path to the hdf5 file referring to the data
-o (optional, default contrastive_file.json in the aprent folder of the hdf5 file): 
    output path with filename for the json file

Created by Mathilde Dupouy, 2024
"""

from pathlib import Path
import pandas as pd
import json

from DataManipulation.hdf5_data import load_from_HDF5
from utils.vocabulary import (
    TRAIN, TEST, DATASET_NAME,
    ID, HARD_POS, PNG_PATH
    )

def generate_contrastive_file(hdf5_path, output_path):
    """
    Generate a JSON file with positives from a dataset hdf5 file,
    with the following structure :
    [subject id] :
            [label] : list of string that are the paths to png images

    Arguments:
    ----------
        hdf5_path: str or Path
        Path to the HDF5 file representing the dataset (see hdf5_data.py for generation and loading)

        output_path: str or Path
        Path where the output file will be created.

    Returns:
    ----------
    """
    # Get data and convert to pandas dataframe
    hdf5_path = Path(hdf5_path)
    hdf5_dict = load_from_HDF5(hdf5_path)
    train_data, test_data = hdf5_dict[DATASET_NAME][TRAIN], hdf5_dict[DATASET_NAME][TEST]
    train_df, test_df = pd.DataFrame(train_data).transpose(), pd.DataFrame(test_data).transpose()

    result_dict = {TRAIN: {}, TEST: {}}
    if len(train_df) > 0:
        for (sub_id, hard_pos), group in train_df.groupby(by=[ID, HARD_POS]) :
            if sub_id not in result_dict[TRAIN].keys():
                result_dict[TRAIN][sub_id] = {}
            result_dict[TRAIN][sub_id][hard_pos] = group[PNG_PATH].tolist()
    if len(test_df) > 0:
        for (sub_id, hard_pos), group in test_df.groupby(by=[ID, HARD_POS]) :
            if sub_id not in result_dict[TEST].keys():
                result_dict[TEST][sub_id] = {}
            result_dict[TEST][sub_id][hard_pos] = group[PNG_PATH].tolist()
    with Path(output_path).open("w") as f:
        json.dump(result_dict, f, indent = 4)

###############################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate contrastive file help")
    parser.add_argument("-p",  "--hdf5_path", type=str, required=True,
                        help="Path to the hdf5 file referring to the data")
    parser.add_argument("-o",  "--output_path", type=str, required=False,
                        help="Path to store the output file")
    
    args = parser.parse_args()
    hdf5_path = Path(args.hdf5_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = hdf5_path.parent/"contrastive_file.json"
    
    generate_contrastive_file(hdf5_path, output_path)
    print("[contrastive_file] File created at", output_path)
    