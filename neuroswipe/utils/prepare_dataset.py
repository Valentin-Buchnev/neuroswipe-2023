import os
import pickle as pkl

import fasttext.util
import jsonlines
import numpy as np
from tqdm import tqdm

from neuroswipe.data_processor import DataProcessor
from neuroswipe.dataloaders import DATASETS_INFO


def prepare_dataset(n_examples_per_file=20, trainval_part=0.05):
    embedding_model = fasttext.load_model(DATASETS_INFO["embedding_model"])

    train_inputs_path = os.path.join(DATASETS_INFO["neuroswipe_train"], "inputs")
    train_targets_path = os.path.join(DATASETS_INFO["neuroswipe_train"], "targets")
    trainval_inputs_path = os.path.join(DATASETS_INFO["neuroswipe_trainval"], "inputs")
    trainval_targets_path = os.path.join(DATASETS_INFO["neuroswipe_trainval"], "targets")

    for p in [train_inputs_path, train_targets_path, trainval_inputs_path, trainval_targets_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    xs = []
    ys = []
    file_number = 0
    total_length = 6e6

    with jsonlines.open(DATASETS_INFO["neuroswipe_train_raw"]) as f:
        for line in tqdm(f.iter(), total=total_length):
            x = DataProcessor().get_processed_item(line)
            y = embedding_model.get_word_vector(line["word"])
            xs.append(x.tolist())
            ys.append(y.tolist())
            if len(xs) == n_examples_per_file:
                for path, arr in zip([train_inputs_path, train_targets_path], [xs, ys]):
                    with open(os.path.join(path, f"{file_number:03d}.pkl"), "wb") as fileObject:
                        pkl.dump(np.array(arr, dtype=np.float32), fileObject)
                xs, ys = [], []
                file_number += 1

    if len(xs) == n_examples_per_file:
        for path, arr in zip([train_inputs_path, train_targets_path], [xs, ys]):
            with open(os.path.join(path, f"{file_number:03d}.pkl"), "wb") as fileObject:
                pkl.dump(np.array(arr, dtype=np.float32), fileObject)

    trainval_length = round(file_number * trainval_part)

    for path_from, path_to in [(train_inputs_path, trainval_inputs_path), (train_targets_path, trainval_targets_path)]:
        for file in os.listdir(path_from):
            if int(file.split(".")[0]) >= file_number - trainval_length:
                src = os.path.join(path_from, file)
                dst = os.path.join(path_to, file)
                os.replace(src, dst)
