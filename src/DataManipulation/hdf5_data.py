#!/usr/bin/env python3
"""
    Code allowing to generate and/or load an hdf5 file for HITS datasets that can be then
    used to train machine/deep learning models.
    It also allows to describe the dataset with respect to a given class in a csv file.

    Created by Yamil Vindas, 2023
    Modified by Mathilde Dupouy, 2024
"""
# Imports
import h5py
from pathlib import Path
import csv

from DataManipulation.explore_dataset import load_info_folder, hms2sec, get_hard_label
from utils.vocabulary import *

def dataset_hdf5_generation(
                        dataset_folder,
                        test_subjects=None,
                        participants_file=None,
                        remove_subjects = []
                    ):
    """
    Creates a hdf5 file called "data.hdf5" containing the split of the data
    between train and test. The structure of the HDF5 file is the following:
        Split (train or test)
            Patient ID
                sample specific ID
                    {
                        [key]: [value] from the id_name_corr if not None
                        [key]: [value] from the participants_file if not None
                        SESSION                         
                        SAMPLE 
                        DETECTION_TIME_HMS
                        DETECTION_TIME_S
                        SOFT_EG
                        SOFT_ES
                        SOFT_A
                        HARD_CLASS                       
                        HARD_POS 
                        EBR                     
                        VEL 
                        LEN                        
                        POS 
                        COEF
                        PRF 
                        INI
                        DURATION
                    }


    Arguments:
    ----------
        dataset_folder: str
            Path to the folder containing the data
        test_subjects: list of str
            List of subjects ids that are going to be used for testing
    """
    print("=======Creating HDF5 file=======")
    # Load dataset informations
    dataset_folder = Path(dataset_folder)
    dataset_dict = load_info_folder(dataset_folder, participants_file=participants_file)

    # Creating an hdf5 file for the dataset
    hdf5_file = h5py.File(dataset_folder/"data.hdf5", "w")
    hdf5_file.create_group(DATASET_NAME)
    
    # Filling the file by mode
    subjects = {
        TRAIN: [sub for sub in dataset_dict.keys() if sub not in test_subjects and sub not in remove_subjects],
        TEST: [sub for sub in test_subjects if sub not in remove_subjects]
    }
    totals = {
        TRAIN: {"A": 0, "ES": 0, "EG": 0, "U": 0, "HITS": 0},
        TEST: {"A": 0, "ES": 0, "EG": 0, "U": 0, "HITS": 0}
    }
    for mode in [TRAIN, TEST]:
        mode_id = 0
        mode_dataset = hdf5_file.create_group(f"{DATASET_NAME}/{mode}")
        for sub in subjects[mode]:
            sub_dict = dataset_dict[sub]
            for ses, ses_dict in sub_dict[SESSIONS].items():
                for run, run_dict in ses_dict.items():
                    sample = mode_dataset.create_group(f"{mode_id}")
                    # Global attributes
                    sample.attrs[ID] = sub
                    sample.attrs[SESSION] = ses
                    sample.attrs[SAMPLE] = run
                    for key, value in {key: value for key, value in sub_dict.items() if key != SESSIONS}.items():
                        sample.attrs[key] = value
                    # Sample attributes
                    sample.attrs[SOFT_A] = run_dict["type"]["A"]
                    sample.attrs[SOFT_EG] = run_dict["type"]["EG"]
                    sample.attrs[SOFT_ES] = run_dict["type"]["ES"]
                    sample.attrs[SOFT_ES] = run_dict["type"]["ES"]
                    sample.attrs[HARD_CLASS] = run_dict[HARD_CLASS]  
                    sample.attrs[HARD_POS] = run_dict[HARD_POS]  
                    pos_keys = [key for key in run_dict.keys() if "pos " in key]
                    for key in pos_keys:
                        sample.attrs[key] = run_dict[key]
                    sample.attrs[PNG_PATH] = str(run_dict[PNG_PATH].resolve())
                    sample.attrs[WAV_PATH] = str(run_dict[WAV_PATH].resolve())
                    sample.attrs[DETECTION_TIME_HMS] = run_dict["detectionTime"]
                    sample.attrs[DETECTION_TIME_S] = hms2sec(run_dict["detectionTime"])
                    sample.attrs[EBR] = run_dict["ebr"]
                    sample.attrs[VEL] = run_dict["vel"]
                    sample.attrs[LEN] = run_dict["len"]
                    sample.attrs[COEF] = run_dict["cof"]

                    totals[mode][run_dict[HARD_CLASS]] += 1
                    totals[mode]["HITS"] += 1
                    mode_id += 1
    
    print(f"{len(subjects[TRAIN])} subjects for training with a total of {totals[TRAIN]['HITS']} HITS (A: {totals[TRAIN]['A']}, ES: {totals[TRAIN]['ES']}, EG: {totals[TRAIN]['EG']}, U: {totals[TRAIN]['U']})")
    print(f"{len(subjects[TEST])} subjects for testing with a total of {totals[TEST]['HITS']} HITS (A: {totals[TEST]['A']}, ES: {totals[TEST]['ES']}, EG: {totals[TEST]['EG']}, U: {totals[TEST]['U']})")
    print("=======HDF5 file created=======")


def load_from_HDF5(hdf5_file_path):
    """
    Create a dictionary containing samples as dictionaries with the following hierarchy:
    dataset name
        mode name
            sample number (str): sample dictionary (cf. dataset_hdf5_generation)


    Arguments:
    ----------
            hdf5_file_path: str
                Path to an hdf5 file containing the structure of the data to use

    Returns:
    --------
        train_data: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the sample
        test_data: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the sample
    """
    print("=======Loading HDF5=======")
    hdf5_obj = h5py.File(hdf5_file_path, 'r')

    hdf5_dict = {} # Dict image of the HDF5 file
    modes = [] # Just for debugging
    for dataset in hdf5_obj:
        hdf5_dict[dataset] = {}
        for mode in hdf5_obj[dataset]:
            hdf5_dict[dataset][mode] = {}
            modes.append(mode)
            for sample in hdf5_obj[dataset][mode]:
                hdf5_dict[dataset][mode][int(sample)] = {}
                for key, value in hdf5_obj[dataset][mode][sample].attrs.items():
                    hdf5_dict[dataset][mode][int(sample)][key] = value
    print("Dataset(s): ", [str(dataset) for dataset in hdf5_obj])
    print("Mode(s): ", modes)
    hdf5_obj.close()

    print(f"=======HDF5 file loaded from {Path(hdf5_file_path).stem}=======")
    return hdf5_dict

def generate_dataset_description(
        hdf5_dict,
        output_path,
        label_name = HARD_CLASS,
        label_keys = ["A", "ES", "EG", "U"],
        subject_key = ID 
    ):
    """
    Describe a dataset from its HDF5 file into a csv file with the following column names :
        ID: subject ID
        [label]: all labels_keys
        "Total"
        "mode": split mode

        and the lines with id "total train", "total test" and "total"
    
    Arguments:
    ----------
        hdf5_dict: dict
            dictionary generated from and HDF5 fils, see structure in load_from_HDF5
        output_path: str or Path
        label_name: str, optional, default HARD_CLASS
            label with respect to which the description is generated
        label_keys: list of str
        subject_key: str, default ID
            Key in the hdf5 file to get the subject id

    Returns:
    --------
    """

    dataset_name = list(hdf5_dict.keys())[0]
    splits = list(hdf5_dict[dataset_name].keys())

    subjects = {mode:[] for mode in splits}
    description_keys = label_keys + ["Total"]
    totals = {
        mode: {key: 0 for key in description_keys} for mode in splits
    }
    dataset_description = {}
    for mode in splits:
        for elem in hdf5_dict[dataset_name][mode].values():
            if elem[subject_key] not in subjects[mode]:
                dataset_description[elem[subject_key]] = {key: 0 for key in description_keys}
                subjects[mode].append(elem[subject_key])
            if subject_key == "PatientID" and label_name == hard_label: # Treat old hdf5 files
                hard_label = get_hard_label({'A':elem["ArtefactScore"], 'EG':elem["EgScore"], 'ES':elem["EsScore"]})
            else:
                hard_label = elem[label_name]
            dataset_description[elem[subject_key]][hard_label] += 1
            dataset_description[elem[subject_key]]["Total"] += 1
            dataset_description[elem[subject_key]]["mode"] = mode
            totals[mode][hard_label] += 1
            totals[mode]["Total"] += 1
        print(f"[hdf5_data] {mode} subjects from hdf5 file :", sorted(list(set(subjects[mode]))))
        print("[hdf5_data] Distribution of classes :", totals[mode])

    dataset_to_csv = [{ID: sub, **dataset_description[sub]} for sub in sorted(list(dataset_description.keys()), key = lambda x : x)]
    for mode in splits:
        dataset_to_csv.append({ID:f"total {mode}", **{key:totals[mode][key] for key in description_keys}, "mode": mode})
    dataset_to_csv.append({ID:"total", **{key:sum([totals[mode][key] for mode in splits]) for key in description_keys}})

    with open(output_path, "w") as f:
        print(output_path)
        dict_writer = csv.DictWriter(f, [ID] + description_keys + ["mode"])
        dict_writer.writeheader()
        dict_writer.writerows(dataset_to_csv)


###############################################################################
###############################################################################

def main():
    """
    Generate an HDF5 file with the parameters test_sub and remove sub if load is true,
    and load this HDF5 file the generate the dataset with respect to class and position.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset HDF5 file help")
    parser.add_argument("-p",  "--dataset_path", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("-t",  "--test_sub", type=str, nargs="+",required=False, default="",
                        help="Subjects ids for test split")
    parser.add_argument("-r",  "--remove_sub", type=str, nargs="+",required=False, default="",
                        help="Subjects ids to remove")
    parser.add_argument("-l",  "--load",  action="store_true", required=False, default=False,
                        help="If the file needs only to be loaded")
    parser.add_argument("-g",  "--generate",  action="store_true", required=False, default=False,
                        help="If a dataset description needs to be generated")
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    
    if not args.load:
        participants_file = dataset_path/"participants.csv"
        test_subjects = args.test_sub
        print("[hdf5_data] Test subjects :", test_subjects)
        remove_subjects = args.remove_sub
        print("[hdf5_data] Remove subjects :", remove_subjects)
        dataset_hdf5_generation(dataset_folder=dataset_path, test_subjects=test_subjects, participants_file=participants_file, remove_subjects=remove_subjects)

    hdf5_path = dataset_path/"data.hdf5"
    assert hdf5_path.exists()
    hdf5_dict = load_from_HDF5(hdf5_path)
    if args.generate:
        generate_dataset_description(hdf5_dict, dataset_path/"description_class.csv")
        generate_dataset_description(hdf5_dict, dataset_path/"description_pos.csv", HARD_POS, ["1", "2", "3", "4", "-1"])

if __name__=="__main__":
    main()
