import fasttext.util
import jsonlines
from tqdm import tqdm

from neuroswipe.data_processor import DataProcessor
from neuroswipe.dataloaders import DATASETS_INFO


def prepare_voc_embeddings():
    embedding_model = fasttext.load_model(DATASETS_INFO["embedding_model"])
    data_processor = DataProcessor()

    with open(DATASETS_INFO["neuroswipe_voc_train"]) as voc_file:
        voc_words = [line.rstrip() for line in voc_file]

    example_grid = {}
    with jsonlines.open(DATASETS_INFO["neuroswipe_train_raw"]) as data_file:
        for example in data_file.iter():
            data_processor.vectorize_data(example)
            data_processor.normalize_data(example)
            example_grid[example["curve"]["grid"]["grid_name"]] = example
            if len(example_grid.keys()) == 2:
                break

    voc_embeddings = {"default": {}, "extra": {}}

    for w in tqdm(voc_words, total=len(voc_words)):
        for grid_type in example_grid.keys():
            w_embedding = embedding_model.get_word_vector(w).flatten()
            voc_embeddings[grid_type][w] = {}
            voc_embeddings[grid_type][w]["embedding"] = w_embedding.tolist()
            voc_embeddings[grid_type][w]["x_begin"] = data_processor.get_char_coords(example_grid[grid_type], w[0])[0]
    with jsonlines.open(DATASETS_INFO["neuroswipe_voc_embedding"], "w") as f:
        f.write(voc_embeddings)
