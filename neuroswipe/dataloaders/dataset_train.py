import os
import pickle

import torch.utils.data as data


class DatasetTrain(data.Dataset):
    def __init__(self, path):
        self.inputs_path = os.path.join(path, "inputs")
        self.targets_path = os.path.join(path, "targets")
        self.inputs_file_names = sorted(list(os.listdir(self.inputs_path)))
        self.targets_file_names = sorted(list(os.listdir(self.inputs_path)))

        self.len = len(self.inputs_file_names)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with open(os.path.join(self.inputs_path, self.inputs_file_names[idx]), "rb") as fileObject:
            xs = pickle.load(fileObject)
        with open(os.path.join(self.targets_path, self.targets_file_names[idx]), "rb") as fileObject:
            ys = pickle.load(fileObject)
        return xs, ys
