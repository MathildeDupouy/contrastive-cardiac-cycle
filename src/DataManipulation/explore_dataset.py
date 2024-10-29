"""
This script contains useful functions like functions to get paths in the dataset or to convert hms format to seconds.
The main function load_info_folder enables to load all informations (metadata + paths) of a dataset in Bids format in a dictionary.

By Mathilde Dupouy, 2024
"""
import os
from pathlib import Path
import matplotlib.image as mpli
import json
import pandas as pd

from utils.vocabulary import *

# id-name CSV keys
CSV_ID = "subjectID"
CSV_NAME = "subjectName"
CSV_SRC = "src"
CSV_PATHOLOGY = "pathology"
CSV_PROCEDURE = "procedure"

def get_vignette_array(vignette_path) :
    """
    Returns a numpy array of the image.
    Arguments:
    ----------
        vignette path : str or Path
        Path to the vignette

    Returns:
    ----------
        numpy.array : vignette array with size :
            (M, N) for grayscale images.
            (M, N, 3) for RGB images.
            (M, N, 4) for RGBA images.

    """
    img = mpli.imread(vignette_path)
    return img

def hms2sec(t_hms) :
    """
    Convert a time in format hhmmssddd where h are hour digits, m minute digits, s second digits and d millisecond digits
    to a time in seconds.
    Arguments:
    ----------
        t_hms : str

    Returns:
    ----------
        str
    """
    t_s = int(t_hms[:2]) * 3600 + int(t_hms[2:4]) * 60 + int(t_hms[4:6]) + int(t_hms[6:9]) * 0.001
    return t_s

def get_hard_label(score_dict, undefined_key = "U"):
    """
    Return a hard label from a score dict. The hard label is a key of the dict or undefined_key if two keys share the same maximum score.
    Arguments:
    ----------
        score_dict : dict {[label] : score}
            Dictionary of the labels scores
        undefined_key: str, optional, default "U"
            String for the undefined label name

    Returns:
    ----------
        str: hard label
    """
    hard_score = max(score_dict.values())
    hard_labels = [key for key, item in score_dict.items() if item == hard_score]
    if len(hard_labels) > 1:
        return undefined_key
    return hard_labels[0]

def load_info_folder(dataset_folder, id_name_corr = None, participants_file = None) :
    """
    Loads the information from a dataset (and associated subject names).
    
    Arguments:
    ----------
    dataset_folder: str or pathlib.Path
        Path to the Bids dataset folder
    id_name_corr: str or pathlib.Path
        Path to a CSV containing the associations between ids and names.
        Default to None to work with entirely anonymous databases.
    participants_file: str or pathlib.Path
        Path to a CSV containing all the informations about a subject.
        Default to None if these informations are not known.

    Returns:
    -------- 
        A dictionary with the following structure: 
        [subject id]: 
            [key]: [value] from the id_name_corr if not None
            [key]: [value] from the participants_file if not None
            HITS: int, number of HITS
            sessions: dict
                [session id]: dict
                    [run id]: dict
                        [key]: [value] from the JSON files of the runs in the dataset
                        pngPath: str or Path
                        wavPath: str or Path

    """
    dataset_folder = Path(dataset_folder)
    assert dataset_folder.is_dir()

    if id_name_corr is not None:
        idNameCorr_df = pd.read_csv(id_name_corr)
        subject_ids = list(idNameCorr_df[CSV_ID])
    else:
        subject_ids = [folder_path.split('sub-')[1] for folder_path in os.listdir(dataset_folder) if (dataset_folder/folder_path).is_dir()]

    if participants_file is not None:
        participants_df = pd.read_csv(participants_file)
        participants_df[CSV_ID] = participants_df[CSV_ID].astype(str)

    info_dict = {}
    for i, subId in enumerate(subject_ids) :
        sub_path = dataset_folder/f"sub-{subId}"
        if sub_path.is_dir() :
            # General information about the subject
            info_dict[subId] = {
                SESSIONS : {},
                }
            if id_name_corr is not None:
                info_dict[subId][SUB_NAME] = idNameCorr_df.loc[idNameCorr_df[CSV_ID] == subId][CSV_NAME].max()
                info_dict[subId][SRC] = idNameCorr_df.loc[idNameCorr_df[CSV_ID] == subId][CSV_SRC].max()
                info_dict[subId][PATHOLOGY] = idNameCorr_df.loc[idNameCorr_df[CSV_ID] == subId][CSV_PATHOLOGY].max()
            if participants_file is not None:
                info_dict[subId][SRC] = participants_df.loc[participants_df[CSV_ID] == subId][CSV_SRC].max()
                info_dict[subId][PATHOLOGY] = participants_df.loc[participants_df[CSV_ID] == subId][CSV_PATHOLOGY].max()
                info_dict[subId][AGE] = participants_df.loc[participants_df[CSV_ID] == subId][AGE].max()
                info_dict[subId][SEX] = participants_df.loc[participants_df[CSV_ID] == subId][SEX].max()
                info_dict[subId][MODALITY] = participants_df.loc[participants_df[CSV_ID] == subId][MODALITY].max()
                info_dict[subId][SERVICE] = participants_df.loc[participants_df[CSV_ID] == subId][SERVICE].max()
                info_dict[subId][MODALITY] = participants_df.loc[participants_df[CSV_ID] == subId][MODALITY].max()
                info_dict[subId][CONTRAST] = participants_df.loc[participants_df[CSV_ID] == subId][CONTRAST].max()
                info_dict[subId][PROCEDURE] = participants_df.loc[participants_df[CSV_ID] == subId][CSV_PROCEDURE].max()
                info_dict[subId][FREQUENCY] = participants_df.loc[participants_df[CSV_ID] == subId][FREQUENCY].max()
                info_dict[subId][DEVICE] = participants_df.loc[participants_df[CSV_ID] == subId][DEVICE].max()

            number_hits = {NB_A: 0, NB_EG: 0, NB_ES: 0, NB_U: 0, NB_HITS: 0}
            for session in os.listdir(sub_path) :
                session_path = sub_path/session
                if session_path.is_dir() :
                    session_number = session.split("ses-")[1]
                    info_dict[subId][SESSIONS][session_number] = {}
                    for sample in os.listdir(session_path) :
                        sample_path = session_path/sample
                        run_number = sample.split('run-')[1]
                        # Run information
                        json_path = sample_path/(sample + '.json')
                        with open(json_path, 'r') as json_file:
                            sample_dict = json.load(json_file)

                        info_dict[subId][SESSIONS][session_number][run_number] = sample_dict
                        if "type" in sample_dict.keys() :
                            hard_label = get_hard_label(sample_dict["type"])
                            info_dict[subId][SESSIONS][session_number][run_number][HARD_CLASS] = hard_label
                        if "pos" in sample_dict.keys() :
                            if type(sample_dict["pos"]) == dict:
                                for pos_key, pos_value in sample_dict["pos"].items():
                                    info_dict[subId][SESSIONS][session_number][run_number][f"pos {pos_key}"]= pos_value
                                hard_pos = get_hard_label(sample_dict["pos"], undefined_key="-1")
                                info_dict[subId][SESSIONS][session_number][run_number][HARD_POS] = hard_pos
                            else:
                                print("HEEERE",subId, session, sample)
                        info_dict[subId][SESSIONS][session_number][run_number][PNG_PATH] = sample_path/(sample + '.PNG')
                        info_dict[subId][SESSIONS][session_number][run_number][WAV_PATH] = sample_path/(sample + '.WAV')
                        number_hits[hard_label] += 1
                        number_hits[NB_HITS] += 1
            info_dict[subId][NB_A] = number_hits[NB_A]
            info_dict[subId][NB_ES] = number_hits[NB_ES]
            info_dict[subId][NB_EG] = number_hits[NB_EG]
            info_dict[subId][NB_U] = number_hits[NB_U]
            info_dict[subId][NB_HITS] = number_hits[NB_HITS]

    return info_dict

if __name__ == "__main__":
    # Test get_hard_label
    test_dict = {"A": 0.4, "ES": 0.9, "EG": 0.4}

    print("Test hard label", get_hard_label(test_dict))

    # Test load_info_folder
    dataset_path = Path("data/Dataset_Bids_1s_reference_posAnnotated_tiny")
    participants_file = dataset_path/"participants.csv"
    info_dict = load_info_folder(dataset_path, participants_file=participants_file)

    sub = "1"
    print(f"Test load_info_folder for subject {sub}")
    print("Info dict keys :", info_dict.keys())
    print("Subject keys :", info_dict[sub].keys())
    print("Number of artefacts :", info_dict[sub]["A"])
    print("Number of solid emboli :", info_dict[sub]["ES"])
    print("Number of gaseous emboli :", info_dict[sub]["EG"])
    print("Number of unknown class HITS :", info_dict[sub]["U"])
    print("Number of HITS :", info_dict[sub]["HITS"])

    print("Procedure : ", info_dict[sub][PROCEDURE]) # only if participants_file is known