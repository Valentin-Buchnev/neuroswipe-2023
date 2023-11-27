from functools import partial
from multiprocessing import Pool

import jsonlines
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neuroswipe.data_processor import DataProcessor
from neuroswipe.dataloaders import DATASETS_INFO, DatasetValidation


def process_func(sample, voc_embeddings, X_EPS=0.1):
    data_processor = DataProcessor()
    idx, x, x_embedding, grid_name = sample

    left_idx = data_processor._bin_search_x(voc_embeddings[grid_name], x[0][0] - X_EPS)
    right_idx = data_processor._bin_search_x(voc_embeddings[grid_name], x[0][0] + X_EPS)
    losses = [(1e9, "a"), (1e9, "a"), (1e9, "a"), (1e9, "a")]
    for i in range(left_idx, right_idx):
        w, info = voc_embeddings[grid_name][i]
        cur_loss = torch.nn.CosineEmbeddingLoss()(
            torch.tensor(np.expand_dims(x_embedding, axis=0)),
            torch.tensor(np.expand_dims(info["embedding"], axis=0)),
            target=torch.tensor([1]),
        )

        if len(losses) == 4 and cur_loss > losses[-1][0]:
            continue
        losses.append((cur_loss, w))
        losses = sorted(losses)
        if len(losses) > 4:
            losses = losses[:4]
    return idx, [l[1] for l in losses]


def prepare_prediction(save_path, model=None, data_type="test", parallel=True, num_workers=60):
    with jsonlines.open(DATASETS_INFO["neuroswipe_voc_embedding"], "r") as f:
        for line in f.iter():
            voc_embeddings = line

    for grid_type in ["default", "extra"]:
        voc_embeddings[grid_type] = [(k, v) for k, v in voc_embeddings[grid_type].items()]
        voc_embeddings[grid_type] = sorted(voc_embeddings[grid_type], key=lambda x: x[1]["x_begin"])

    val_data = DatasetValidation(model, data_type=data_type)
    results = [None] * len(val_data)

    if parallel:
        with Pool(num_workers) as executor:
            func = partial(process_func, voc_embeddings=voc_embeddings)
            for idx, result in tqdm(
                executor.imap_unordered(func, val_data, chunksize=len(val_data) // (num_workers * 2)),
                total=len(val_data),
            ):
                results[idx] = result
    else:
        func = partial(process_func, voc_embeddings=voc_embeddings)
        for idx, result in tqdm(map(func, val_data), total=len(val_data)):
            results[idx] = result

    pd.DataFrame(results).to_csv(save_path, header=False, index=False)
