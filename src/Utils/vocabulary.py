"""
Define constant strings for the application.
"""
#--------- GENERAL
TOTAL = "total"

# Splits
TRAIN = "train"
VAL = "validation"
TEST = "test"
ALL = "all"

# Datasets
INDEX = "dataset index"

# Prediction
PREDICTED = "predicted"

#--------- CONFIG KEYS
# General
NAME = "name"
DESCRIPTION = "description"
INFO = "info"

# Training
DEVICE = "device"
NB_EPOCHS = "nb_epochs"
CHECKPOINT_FREQ = "checkpoint frequency"

# Model
MODEL = "model"
ARGS = "args"

# Optimization
OPTIMIZER = "optimizer"
LR = "lr"
BATCH_SIZE = "batch size"
WEIGHT_DECAY = "weight_decay"
#SCHEDULER

# Data
DATASET = "dataset"
DATA_PATH = "data path"
DATASET_TYPE = "dataset type"
TRAIN_LABELS = "train labels"
TEST_LABELS = "test labels"
#AUG
DATALOADER = "dataloader"

# Loss and mertics
LOSS = "loss"
METRIC = "metric"
WEIGHT = "weight"
PRED_POS = "prediction position"

# Evaluation
EVALUATION = "evaluation"
LABELS = "labels"

# Representation
REPRESENTATION = "representation"

# Projection
PROJECTION = "projection"


#--------- HITS MANAGEMENT
# General
PARTICIPANTS_PATH = "participants path"

# HDF5 file
DATASET_NAME = "mydataset"
# Subject metadata
ID = "ID"
SUB_NAME = "Name"
SRC = "source_id"
SRC_H = "source"
PATHOLOGY = "pathology_id"
PATHOLOGY_H = "pathology"
SESSIONS = "sessions"
NB_HITS = "HITS"
NB_A = "A"
NB_EG = "EG"
NB_ES = "ES"
NB_U = "U"
# Participants CSV keys
AGE = "age"
SEX = "sex"
MODALITY = "modality"
SERVICE = "service"
PROCEDURE = "procedure_id"
PROCEDURE_H = "procedure"
CONTRAST = "contrast"
FREQUENCY = "frequency"
DEVICE = "device"
# Sample metadata
SESSION = "session"
SAMPLE = "sample number"
PNG_PATH = "pngPath"
WAV_PATH = "wavPath"
DETECTION_TIME_HMS = "detection time (hh:mm:ss:xxx)"
DETECTION_TIME_S = "detection time (s)"
SOFT_EG = "EG probability"
SOFT_ES = "ES probability"
SOFT_A = "A probability"
EBR = "ebr"
VEL = "velocity"
LEN = "length"
POS = "position"
COEF = "coefficient"
PRF = "prf (Hz)"
INI = "initial time (s)"
DURATION = "duration (s)"

# Labels
CONTRASTIVE = "contrastive"
UNSUPERVISED = "unsupervised"
HARD_CLASS = "class"
HARD_POS = "hard pos"
SOFT_CLASS = "soft class"
SOFT_POS = "soft pos"

class_index = {
                "ES": 0,
                "EG": 1,
                "A": 2,
                "U": -1
            }

pos_index = {
                '1': 0,
                '2': 1,
                '3': 2,
                '4': 3,
                '-1': 4
            }

index2class = {value:key for key, value in class_index.items()}
index2pos = {value:key for key, value in pos_index.items()}

# Contrastive info
CONTRASTIVE_PATH = "contrative file path"
POS_PATH = "positive png path"
NEG_PATH = "negative png path"