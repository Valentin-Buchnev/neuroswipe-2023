import json

import numpy as np
import torch
from tqdm import tqdm

from neuroswipe.data_processor import DataProcessor
from neuroswipe.dataloaders import DATASETS_INFO


class DatasetValidation:
    def __init__(self, model, data_type="test"):
        self.data = []
        data_processor = DataProcessor()

        assert data_type in ["valid", "test"]
        path = DATASETS_INFO["neuroswipe_valid_raw"] if data_type == "valid" else DATASETS_INFO["neuroswipe_test_raw"]
        with open(path, "r") as file:
            for i, line in tqdm(enumerate(file)):
                sample = json.loads(line)
                x = data_processor.get_processed_item(sample)
                model = model.to("cuda:0")
                model.eval()
                x = torch.tensor(x.astype(np.float32)).unsqueeze(0).to("cuda:0")
                with torch.no_grad():
                    embedding = model(x)
                embedding = embedding.reshape(-1).detach().cpu().numpy()
                x = x.squeeze(0).detach().cpu().numpy()
                self.data.append((i, x, embedding, sample["curve"]["grid"]["grid_name"]))
        self.data = sorted(self.data, key=lambda x: x[1][0][0])
        model = model.to("cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
