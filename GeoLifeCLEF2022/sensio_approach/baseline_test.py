import os
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl

from src import RGBDataset, RGBDataModule, RGBModule

if __name__ == "__main__":
    path = Path("../data")
    os.listdir(path)

    dm = RGBDataModule()
    dm.setup()

    imgs, labels = next(iter(dm.train_dataloader(batch_size=25)))

    hparams = {
        "datamodule": {
            "batch_size": 256,
            "num_workers": 4,
            "pin_memory": True,
        },  
        "backbone": "resnet18",
        "pretrained": True,
        "optimizer": "Adam",
        "optimizer_params": {
            "lr": 1e-3,
        }
    }

    dm = RGBDataModule(**hparams["datamodule"])
    module = RGBModule(hparams)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        enable_checkpointing=False,
        logger=None,
        #overfit_batches=1,
    )

    trainer.fit(module, dm)