from pathlib import Path
import json
import numpy as np
import torch

from DataManipulation.HITS_2D_dataset import HITS_2D_Dataset
from Utils.vocabulary import (
    ID,
    HARD_CLASS, HARD_POS,
    POS_PATH, NEG_PATH, PNG_PATH
)

class HITS_contrastive_Dataset(HITS_2D_Dataset):
    """A HITS_2D_Dataset with a dictionary of positives for contrastive experiments."""
    def __init__(self, data, contrastive_groups, nb_pos = 1, nb_neg = 1, neg_intrasub = False, neg_samepos = True, duration = None, initial_duration = 1000, multiple = None, label_keys = [HARD_CLASS], remove_subjects = [],remove_pos_U = True, remove_class_U = True) -> None:
        """
        Initialize the HITS_2D_Dataset, get the positive groups from the contrastive file,
        and remove the sample with less than nb_pos + 1 elements.
        Arguments:
        ----------
            data: dict
            A dictionary of samples.

            contrastive_groups: dict{[subject id]: dict{[hard position label]: list of str of png paths}}
                Dictionaries of groups of png paths to vignettes

            nb_pos: int, default 1
                Number of positives

            nb_neg: int, default 1
                Number of negatives
            
            neg_intrasub: boolean, default False
                if True, negatives are only taken from the same subject as the anchor

            neg_samepos: boolean, default True
                if True, negatives can be taken at the same position as the anchor
            
            label_keys: list of str
            The keys that have to be extracted from data.

            duration: int, default None
            The desired duration of vignettes, in milliseconds. If None, duration is the initial duration

            initial_duration: int
            The initial duration of the vignettes in data, in milliseconds.

            multiple: int, default None
            Resize vignettes so that height and width are multiple of multiple.

            remove_pos_U: bool, default True
            If True, the samples with an unknown position (non nul score in -1) are removed

            remove_class_U: bool, default True
            If True, the samples with an unknown class (non unique argmax) are removed
        """
        # Initialise the HITS 2D dataset
        super().__init__(data, duration, initial_duration, multiple, label_keys, remove_subjects, remove_pos_U, remove_class_U)

        self.nb_pos = nb_pos
        self.nb_neg = nb_neg

        self.neg_intrasub = neg_intrasub
        self.neg_samepos = neg_samepos

        self.groups = contrastive_groups
        for subid in self.remove_subjects:
            self.groups.pop(subid, None)
        
        # Removing samples that have no pairs
        print("[HITS contrastive dataset] Initial length", len(self.data))
        data_copy = self.data[:]
        for sample in data_copy:
            if sample[ID] not in self.groups.keys():
                self.data.remove(sample)
            elif sample[HARD_POS] not in self.groups[sample[ID]].keys():
                self.data.remove(sample)
            elif len(self.groups[sample[ID]][sample[HARD_POS]]) < nb_pos + 1:
                # remove groups with not enough positives
                self.data.remove(sample)
        print("[HITS contrastive dataset] Final length", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get anchor
        x, label, info = super().__getitem__(index)

        # Draw positive(s)
        x_pos = []
        positive_paths = np.random.choice([sample for sample in self.groups[info[ID]][info[HARD_POS]] if sample != info[PNG_PATH]], 
                                          size = self.nb_pos, replace = False)
        x_pos = torch.Tensor(np.array([self.get_array_from_path(positive_path) for positive_path in positive_paths]))
        info[POS_PATH] = list(positive_paths)

        # Draw negative(s)
        # negative_list = [self.groups[neg_sub][neg_label] for neg_label in self.groups[neg_sub].keys() for neg_sub in self.groups.keys() 
        #                  if (not self.neg_samepos and neg_label == info[HARD_POS]) or (not self.neg_intrasub and neg_sub == info[ID]) or (self.neg_intrasub and neg_sub == info[ID] and neg_label == info[HARD_POS])]
        x_neg = []
        info[NEG_PATH] = []
        for _ in range(self.nb_neg):
            neg_sub = np.random.choice(list(self.groups.keys()), size = 1)[0]
            neg_label = np.random.choice(list(self.groups[neg_sub].keys()), size = 1)[0]
            while ((not self.neg_samepos and neg_label == info[HARD_POS]) 
                   or (not self.neg_intrasub and neg_sub == info[ID]) 
                   or (self.neg_intrasub and neg_sub == info[ID] and neg_label == info[HARD_POS])) :
                neg_sub = np.random.choice(list(self.groups.keys()), size = 1)[0]
                neg_label = np.random.choice(list(self.groups[neg_sub].keys()), size = 1)[0]
            negative_path = np.random.choice(self.groups[neg_sub][neg_label], size = 1)[0]
            x_neg.append(self.get_array_from_path(negative_path))
            info[NEG_PATH].append(negative_path)
        x_neg = torch.Tensor(np.array(x_neg))
        info[NEG_PATH] = info[NEG_PATH]

        return (x, x_pos, x_neg), label, info
