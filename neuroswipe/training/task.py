import pytorch_lightning as pl
import torch

from neuroswipe.callbacks import ModelSaverCallback
from neuroswipe.models import create_model


class Task:
    def __init__(
        self,
        training_loop,
        epochs,
        model_saver,
        model_arch=None,
        optimizers=None,
        callbacks=None,
    ):
        self.training_loop = training_loop
        self.epochs = epochs
        self.model_saver = model_saver

        self.model_arch = model_arch

        self.optimizers = optimizers
        self.callbacks = callbacks or []

    def set_model(self):
        if not self.model_arch:
            raise RuntimeError(
                "Please, provide the model arch in the task constructor (or in a yaml configuration file)."
            )
        model = create_model(self.model_arch)

        self.training_loop.set_model(model)

    def set_optimizers(self):
        optimizers = []
        schedulers = []
        for _, opt in self.optimizers.items():
            optimizer_class = getattr(torch.optim, opt["name"])
            optimizer = optimizer_class(params=self.training_loop.model.parameters(), **opt["params"])

            scheduler_class = getattr(torch.optim.lr_scheduler, opt["scheduler"]["name"])
            scheduler = scheduler_class(optimizer, **opt["scheduler"]["params"])

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        self.training_loop.set_optimizers(optimizers=optimizers, schedulers=schedulers)

    def configure(self, logdir=None, num_gpus=1):

        self.set_model()
        self.set_optimizers()

        # The batch size comes from config where we expect the effective batch size,
        # whereas pytorch lightning expects the batch size per gpu.
        self.training_loop.dataset.batch_size //= num_gpus

        # logger
        logger = pl.loggers.TensorBoardLogger(logdir, name=None)

        # callbacks
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        model_saver = ModelSaverCallback(logdir, **self.model_saver)
        progress_bar = pl.callbacks.RichProgressBar()

        # pl.Trainer
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=num_gpus,
            max_epochs=self.epochs,
            logger=logger,
            callbacks=[lr_monitor, model_saver, progress_bar] + self.callbacks,
        )

    def run(self):
        self.trainer.fit(
            self.training_loop,
            self.training_loop.dataset.train_loader,
            self.training_loop.dataset.val_loader,
        )
