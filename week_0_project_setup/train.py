import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    # Save the best trained model -> choose which model by monitoring a metric
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    # Helps the model not to overfit by monitoring on a certain param
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=("mps" if torch.backends.mps.is_available() else "cpu"),
        max_epochs=5,
        fast_dev_run=False, # run one batch of training step and one batch of val step
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
