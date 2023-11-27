import fasttext.util

from neuroswipe.dataloaders import DATASETS_INFO


def prepare_embedding_model():
    fasttext.util.download_model("ru")
    embedding_model = fasttext.load_model("cc.ru.300.bin")
    fasttext.util.reduce_model(embedding_model, 200)
    embedding_model.save_model(DATASETS_INFO["embedding_model"])
