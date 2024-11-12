import torch
import numpy as np
from torch.utils.data import Dataset

from Utils.tools import get_centered_range, get_closest_multiple
from DataManipulation.explore_dataset import get_vignette_array
from DataManipulation.explore_dataset import HARD_CLASS, HARD_POS, PNG_PATH
from DataManipulation.hdf5_data import SOFT_A, SOFT_EG, SOFT_ES, ID
from Utils.vocabulary import (
    ID,
    INDEX,
    HARD_CLASS, SOFT_CLASS, HARD_POS, SOFT_POS, UNSUPERVISED,
    class_index, pos_index,
    PNG_PATH,
    SOFT_A, SOFT_EG, SOFT_ES,
)

class HITS_2D_Dataset(Dataset):
    """A dataset to store HITS vignettes, their labels and metadata."""
    def __init__(self, data, duration = None, initial_duration = 1000, multiple = None, label_keys = [HARD_CLASS], remove_subjects = [],remove_pos_U = True, remove_class_U = True) -> None:
        """
        Arguments:
        ----------
            data: dict
            A dictionary of samples.

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
        super().__init__()

        print("======Creating dataset======")
        # Data
        self.data = data
        # Labels keys
        self.label_keys = label_keys
        self.remove_pos_U = remove_pos_U
        self.remove_class_U = remove_class_U
        self.remove_subjects = remove_subjects

        # Removing samples with undefined labels (class or position)
        samples_to_remove = []
        if self.remove_class_U:
            for sample, sample_dict in data.items():
                if sample_dict[HARD_CLASS] == "U":
                    samples_to_remove.append(sample)
        nb_removed_class_U = len(samples_to_remove)
        print(f"[HITS dataset] {nb_removed_class_U} samples removed because of an undefined class.")
        if self.remove_pos_U:
            for sample, sample_dict in data.items():
                if sample_dict["pos -1"] > 0 and sample not in samples_to_remove:
                    samples_to_remove.append(sample)
        nb_removed_pos_U = len(samples_to_remove) - nb_removed_class_U
        print(f"[HITS dataset] {nb_removed_pos_U} samples removed because of an undefined position.")
        if len(self.remove_subjects) > 0:
            for sample, sample_dict in data.items():
                if sample_dict[ID] in self.remove_subjects and sample not in samples_to_remove:
                    samples_to_remove.append(sample)
        print(f"[HITS dataset] {len(samples_to_remove) - nb_removed_pos_U - nb_removed_class_U} samples removed because of an undesired subject.")
        for sample in samples_to_remove:
            self.data.pop(sample)
        # Because some indices have been popped, data is stored in a list to have continuous indices
        self.data = [data[id] for id in sorted(list(data.keys()))]

        self.width = None
        self.height = None
        if len(self.data) > 0:
            # Define final size of the vignette
            if duration is None:
                duration = initial_duration
            first_elem = get_vignette_array(self.data[0][PNG_PATH])
            print(f"[HITS dataset] Image initial size {first_elem.shape}")

            # Resize the image to be a multiple of multiple
            if multiple is not None:
                # Height
                self.height = get_closest_multiple(first_elem.shape[0], multiple)
                h_diff = first_elem.shape[0] - self.height
                self.h_start = int(h_diff // 2)
                self.h_stop = int(h_diff // 2 + self.height)
                # Width
                new_width = round(first_elem.shape[1] * duration / initial_duration)
                self.width = get_closest_multiple(new_width, multiple)
                self.duration = initial_duration * self.width / first_elem.shape[1]
                w_diff = first_elem.shape[1] - self.width
                self.w_start = int(w_diff // 2)
                self.w_stop = int(w_diff // 2 + self.width)
                # TODO how the vignette will be centered ? Do we want to be accurate up to a constant?
            else: 
                # Duration can be modified to have a centered vignette
                self.h_start, self.h_stop = 0, first_elem.shape[0]
                self.height = self.h_stop - self.h_start
                self.w_start, self.w_stop, self.duration = get_centered_range(new_duration=duration, old_duration=initial_duration, old_width=first_elem.shape[1])
                self.width = self.w_stop - self.w_start
        else:
            self.duration = duration

        print(f"[HITS dataset] Image final size {self.height, self.width, 3}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item[INDEX] = index
        img = self.get_array_from_path(item[PNG_PATH])

        # Building labels
        labels = {}
        if HARD_CLASS in self.label_keys:
            labels[HARD_CLASS] = class_index[item[HARD_CLASS]]
        if HARD_POS in self.label_keys:
            if str(item[HARD_POS]) in pos_index.keys():
                labels[HARD_POS] = pos_index[str(item[HARD_POS])]
            else:
                labels[HARD_POS] = pos_index["-1"]
        if SOFT_CLASS in self.label_keys:
            label = torch.zeros((3, 1), dtype=torch.float32)
            label[0] = item[SOFT_ES]
            label[1] = item[SOFT_EG]
            label[2] = item[SOFT_A]
            labels[SOFT_CLASS] = label
        if SOFT_POS in self.label_keys:
            if self.remove_class_U:
                label = torch.zeros((4, 1), dtype=torch.float32)
            else:
                label = torch.zeros((5, 1), dtype=torch.float32)
                if "pos -1" in item.keys():
                    label[4] = item["pos -1"]
            if "pos 1" in item.keys() and "pos 2" in item.keys() and "pos 3" in item.keys() and "pos 4" in item.keys() :
                label[0] = item["pos 1"]
                label[1] = item["pos 2"]
                label[2] = item["pos 3"]
                label[3] = item["pos 4"]
            labels[SOFT_POS] = label
        if UNSUPERVISED in self.label_keys:
            labels[UNSUPERVISED] = img

        return img, labels, item
    
    def get_array_from_path(self, png_path):
        """
        Returns a numpy array corresponding to the image at png_path, which is cropped and reshaped 
        to have the final shape (3, self.height, self.width).
        """
        # Getting 2D array
        img = get_vignette_array(png_path)
        # Cropping the alpha channel if loaded
        img = img[:, :, :3]
        # Resizing the image
        img = img[self.h_start:self.h_stop, self.w_start:self.w_stop, :]
        # Reshape (channel, H, W)
        img = np.transpose(img, (2, 0, 1))

        return img
