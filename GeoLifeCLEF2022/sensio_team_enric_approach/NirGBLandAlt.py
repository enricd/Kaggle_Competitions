from src import NirGBLandAltModule, NirGBLandAltDataModule
import pytorch_lightning as pl
import sys
import yaml
from src import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

config = {
    'backbone': 'resnet18',
    'pretrained': True,
    'optimizer': 'Adam',
    'optimizer_params': {
        'lr': 3e-4
    },
    'early_stopping': False,
    'trainer': {
        'gpus': 1,
        'max_epochs': 30,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': True,
        'precision': 16
    },
    'datamodule': {
        'batch_size': 512,
        'num_workers': 8,
        'pin_memory': True
    },
}


def train(config, name):
    pl.seed_everything(42, workers=True)
    dm = NirGBLandAltDataModule(**config['datamodule'])
    module = NirGBLandAltModule(config)
    if config['trainer']['logger']:
        config['trainer']['logger'] = WandbLogger(
            project="GeoLifeCLEF2022",
            name=name,
            config=config
        )
    config['trainer']['callbacks'] = []
    if config['trainer']['enable_checkpointing']:
        config['trainer']['callbacks'] += [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{name}-{{val_loss:.5f}}-{{epoch}}',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            )
        ]
    if config['early_stopping']:
        config['trainer']['callbacks'] += [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=True
            )
        ]
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(module, dm)
    trainer.save_checkpoint('final.ckpt')


if __name__ == '__main__':
    name = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        name = config_file[:-4]
        if config_file:
            with open(config_file, 'r') as stream:
                loaded_config = yaml.safe_load(stream)
            deep_update(config, loaded_config)
    print(config)
    train(config, name)