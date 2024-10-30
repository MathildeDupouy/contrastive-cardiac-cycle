from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

from DataManipulation.projection import get_exp_projection
from Experiments.experiments import get_dataloaders
from Utils.vocabulary import (
    INDEX, PROJECTION,
    ID, SESSION, SAMPLE, DEVICE,
    PNG_PATH, WAV_PATH, 
    SOFT_EG, SOFT_ES, SOFT_A,
    HARD_POS
)

SUB_CSV = "Subject"
SES_CSV = "Session"
RUN_CSV = "Run"
META_CSV = "MetadataFile"
AUDIO_CSV = "AudioFile"
IMAGE_CSV = "ImageFile"
A_CSV = "ArtScore"
GE_CSV = "GEScore"
SE_CSV = "SEScore"
X_CSV = "xCoord"
Y_CSV = "yCoord"
LQ_CSV = "localQuality"
AUT0_CSV = "AutoLabel"
VERIFIED_CSV = "VerifiedSample"
ANNOTATOR_CSV = "Annotator"
VERIFIED_BY_CSV = "VerifiedBy"
POS_CSV = "position"

def prepare_reference_file(exp_path, output_name, projection_name = "projection.pkl", representation_name = "representations.pkl", position=True):
    """
    Prepare a file that can be used in DataAnnotationApp, with information paths and projection.
    This file is extracted from the dataset described in the config JSON and the model last_model.
    Representations are computed only if a representation file is not available in the experiment folder.

    Arguments:
    ----------
        exp_path: str or Path
        Path to the experiment folder.

        output_path: str or Path
        Path to give to the created CSV file.

        position: bool, default True
        If a column for the posiyion is created
    """

    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Path {exp_path} doesn't exist"
    # Getting the configuration
    config_path = exp_path/"config.json"
    assert config_path.exists(),  f"Path {config_path} doesn't exist"
    with config_path.open('r') as f:
        json_config = json.load(f)
    dataloaders, _, _ = get_dataloaders(json_config)
    device = json_config[DEVICE]
    projection_dict = get_exp_projection(exp_path, model = None, dataloaders=dataloaders, device=device, representation_name=representation_name)

    # Creating a DataFrame with data
    csv_df = pd.DataFrame(columns=[
        SUB_CSV, SES_CSV, RUN_CSV, META_CSV, AUDIO_CSV, IMAGE_CSV, 
        A_CSV, GE_CSV, SE_CSV, X_CSV, Y_CSV, LQ_CSV, 
        AUT0_CSV, VERIFIED_CSV, ANNOTATOR_CSV, VERIFIED_BY_CSV
    ])
    for mode in dataloaders.keys():
        mode2 = "validation" if mode == "test" and "test" not in projection_dict.keys() else mode #TODO
        i = 0
        for batch in tqdm(dataloaders[mode]):
            x, labels, info = batch
            batch_size = len(info[ID])
            meta_path = [path.replace(".PNG", ".json") for path in info[PNG_PATH]]
            batch_dict = {
                SUB_CSV: info[ID], SES_CSV: info[SESSION], RUN_CSV: info[SAMPLE],
                META_CSV: meta_path,
                AUDIO_CSV: info[WAV_PATH], IMAGE_CSV: info[PNG_PATH],
                A_CSV: info[SOFT_A], GE_CSV: info[SOFT_EG], SE_CSV: info[SOFT_ES],
                LQ_CSV: 'None', AUT0_CSV: False, VERIFIED_CSV: True, ANNOTATOR_CSV: None, VERIFIED_BY_CSV: None
                }
            
            i_proj = np.zeros_like(info[INDEX])
            if INDEX in projection_dict[mode2].keys():
                for i_idx, batch_idx in enumerate(info[INDEX]):
                    i_proj_idx = np.where(projection_dict[mode2][INDEX] == float(batch_idx))[0]
                    assert len(i_proj_idx) == 1
                    i_proj[i_idx] = i_proj_idx[0]            
            else:
                for i_idx, batch_id in enumerate(info[ID]):
                    i_proj_idx = np.where((np.array(projection_dict[mode2][ID]) == str(batch_id)) & (np.array(projection_dict[mode2][SAMPLE]) == str(info[SAMPLE][i_idx])))[0]
                    # print(batch_id, info[SAMPLE][i_idx], len(i_proj_idx))
                    # print([projection_dict[mode2][PROJECTION][i] for i in i_proj_idx])
                    # assert len(i_proj_idx) == 1
                    i_proj[i_idx] = i_proj_idx[0]
            batch_dict[X_CSV] = list(projection_dict[mode2][PROJECTION][i_proj, 0])
            batch_dict[Y_CSV] = list(projection_dict[mode2][PROJECTION][i_proj, 1])

            if position:
                batch_dict[POS_CSV] = info[HARD_POS]

            csv_df = pd.concat((csv_df, pd.DataFrame(batch_dict)))
            i += batch_size
    # Saving CSV
    output_path = exp_path/output_name
    with output_path.open("w") as f:
        csv_df.to_csv(f, index=False)
        print(f"Projection file saved at {output_path}")

    return csv_df

###############################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment projection help")
    parser.add_argument("-p",  "--exp_path", type=str, nargs="+", required=True,
                        help="Experiment paths to the experiment folders to be projected")
    parser.add_argument("-o",  "--output_name", type=str, required=True,
                        help="Name for the outputted csv")
    parser.add_argument("-pn",  "--projection_name", type=str, required=False, default= "projection.pkl",
                        help="Name of the input projection")
    parser.add_argument("-rn",  "--representation_name", type=str, required=False, default= "representations.pkl",
                        help="Name of the input representation")
    

    args = parser.parse_args()
    exp_paths = args.exp_path

    for exp_path in exp_paths:
        prepare_reference_file(exp_path, args.output_name, args.projection_name if args.projection_name else None, args.representation_name if args.representation_name else None)


