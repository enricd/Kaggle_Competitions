from src import RGBModule, RGBDataModule
import pytorch_lightning as pl
import sys
import yaml
from src import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    "backbone": "resnet18",
    "pretrained": True,
    "optimizer": "Adam",
    "optimizer_params": {
        "lr": 1e-3,
    },
    "trainer": {
        "gpus": 1,
        "max_epochs": 10,
        "enable_checkpointing": False,
        "logger": None,
        "overfit_batches": 0,
    },
    "datamodule": {
        "batch_size": 400,
        "num_workers": 2,
        "pin_memory": False,
    },  
}


def train(config, name):
    module = RGBModule(config)
    dm = RGBDataModule(**config["datamodule"])
    if config["trainer"]["logger"]:
        config["trainer"]["logger"] = WandbLogger(
            project="GeoLifeCLEF2022",
            name=name,
            config=config,
        )
    if config["trainer"]["enable_checkpointing"]:
        config["trainer"]["callbacks"] = [
            ModelCheckpoint(
                dirpath="./checkpoints",
                filename=f"{name}-{{val_loss:.5f}}-{{epoch}}",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )
        ]
    trainer = pl.Trainer(**config["trainer"])
    trainer.fit(module, dm)
    trainer.save_checkpoint("final.ckpt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        name = config_file[:-4]
        if config_file:
            with open(sys.argv[1], "r") as f:
                loaded_config = yaml.safe_load(f)
            deep_update(config, loaded_config)
    print(config)
    train(config, name)
