import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
#from .dm import DataModule
#from .model import Model
from src import DataModule, Model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

config = {
    'lr': 3e-4,
    'optimizer': 'Adam', # new
    'batch_size': 32,   
    'max_epochs': 50,
    'precision': 16,
    'test_size': 0.2,
    'seed': 42,
    'subset': 0.1,
    'size': 256,
    'backbone': 'resnet18',
    'val_batches': 10,
    'extra_data': 0
}

dm = DataModule(
    file = 'train_extra' if config['extra_data'] else 'train_old.csv' 
    **config)

model = Resnet(config)

wandb_logger = WandbLogger(project="cassava", config=config)      # Logging with weights and biases

es = EarlyStopping(monitor='val_acc', mode='max', patience=3)
checkpoint = ModelCheckpoint(
    dirpath='./',
    filename=f'{config["backbone"]}-{config["size"]}-{val_acc:.5f}', 
    save_top_k=1, monitor='val_acc', mode='max')

trainer = pl.Trainer(
    gpus=1, 
    precision=config['precision'],
    logger=wandb_logger,
    max_epochs=config['epochs'],
    #limit_train_batches=5,
    #limit_val_batches=5
    callbacks=[es, checkpoint],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)