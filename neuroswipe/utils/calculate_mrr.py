import pandas as pd
from tqdm import tqdm

from neuroswipe.dataloaders import DATASETS_INFO
from neuroswipe.metrics import MRR


def calculate_mrr(pred_path):
    metric = MRR()

    pred = pd.read_csv(pred_path, header=None)
    gt = pd.read_csv(DATASETS_INFO["neuroswipe_valid_ref"], header=None)

    for i in tqdm(range(len(pred))):
        metric.update(pred.iloc[[i]].to_numpy()[0], gt.iloc[[i]].to_numpy()[0][0])
    print(metric.compute())
