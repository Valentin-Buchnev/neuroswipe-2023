import pytorch_lightning as pl
import torch


class TrainingLoop(pl.LightningModule):
    def __init__(
        self,
        dataset,
        loss=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset

        self.loss = loss

    @property
    def automatic_optimization(self):
        return False

    def set_model(self, model):
        self.model = model

    def set_optimizers(self, optimizers, schedulers):
        self.optimizer = optimizers
        self.scheduler = schedulers

    def configure_optimizers(self):
        return self.optimizer, self.scheduler

    def log_losses(self, prefix="", progress_bar=True):
        for name, value in self.logs.items():
            self.log(prefix + name, value, on_step=False, on_epoch=True, sync_dist=True)
            if progress_bar:
                self.log("pb_" + prefix + name, value, logger=False, on_step=True, prog_bar=True, sync_dist=True)

    def compute_total_loss(self, model_outputs, batch):
        model_outputs = model_outputs.reshape((-1, model_outputs.shape[-1]))
        shape = model_outputs.shape
        loss = self.loss(
            model_outputs,
            self.get_targets(batch).reshape(shape),
            target=torch.ones((shape[0],)).to(model_outputs.device),
        )

        self.logs["target_loss"] = torch.clone(loss)
        return loss

    def optimization_step(self, loss):
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

    def get_model_inputs(self, batch):
        return batch[0].reshape((-1, batch[0].shape[-2], batch[0].shape[-1]))

    def get_targets(self, batch):
        return batch[1].reshape((-1, 1, batch[1].shape[-1]))

    def training_step(self, train_batch, batch_idx, logs=None):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(train_batch))

        loss = self.compute_total_loss(model_outputs, train_batch)

        self.optimization_step(loss)
        self.log_losses(prefix="train_")

        return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, val_batch, batch_idx):
        self.logs = {}
        model_outputs = self.model(self.get_model_inputs(val_batch))

        loss = self.compute_total_loss(model_outputs, val_batch)
        self.log_losses(prefix="val_", progress_bar=False)

        return loss

    def validation_epoch_end(self, outputs):
        pass
