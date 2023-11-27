import jsonlines
from tqdm import tqdm

from neuroswipe.dataloaders import DATASETS_INFO


def prepare_vocabulary():
    unique_words = set()

    with jsonlines.open(DATASETS_INFO["neuroswipe_train_raw"]) as f:
        for line in tqdm(f.iter(), total=6e6):
            unique_words.update({line["word"]})

    with open(DATASETS_INFO["neuroswipe_voc_train"], "w") as f:
        for w in list(unique_words):
            f.write(w + "\n")
