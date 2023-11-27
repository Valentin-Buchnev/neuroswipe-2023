import torch

from neuroswipe.dataloaders.dataset_train import DatasetTrain
from neuroswipe.dataloaders.datasets_info import DATASETS_INFO


class DataloaderTrain:
    def __init__(
        self,
        batch_size,
        num_workers=8,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        if not self._train_loader:
            dataset = DatasetTrain(
                DATASETS_INFO["neuroswipe_train"],
            )

            self._train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        if not self._val_loader:
            dataset = DatasetTrain(
                DATASETS_INFO["neuroswipe_trainval"],
            )

            self._val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._val_loader
