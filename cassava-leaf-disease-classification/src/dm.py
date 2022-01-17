import pytorch_lightning as pl        # previously isntalled with pip install pytorch-lightning
import torch
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path, imgs, labels, trans=None):
        self.path = path
        self.imgs = imgs
        self.labels = labels
        self.trans = trans
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, ix):
        img = torchvision.io.read_image(f'{self.path}/{self.imgs[ix]}').float() / 255.
        if self.trans:
            img = self.trans(img)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label
    

class DataModule(pl.LightningDataModule):

    def __init__(self, 
                 path='data',
                 file='train_old.csv',
                 batch_size=64, 
                 test_size=0.2, 
                 seed=42, 
                 subset=False,
                 size=256,
                **kwargs
    ):
        
        super().__init__()
        self.path = path
        self.file = file
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.subset = subset
        self.size = size

    def setup(self, stage=None):
        # read csv file with imgs names and labels
        df = pd.read_csv(f'{self.path}/{self.file}')
        # split in train / val
        train, val = train_test_split(
            df,
            test_size = self.test_size,
            shuffle = True,
            stratify = df['label'],           # stratified by labels guaranties that labels are balanced in train and test
            random_state = self.seed
        )
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        if self.subset:
            _, train = train_test_split(
                train, 
                test_size = self.subset,
                shuffle = True,
                stratify = train['label'],
                random_state = self.seed
            )            
            print("Training only on ", len(train), "samples.")
            
        # train dataset
        train_trans = torchvision.transforms.CenterCrop(self.size)
        self.train_dataset = Dataset(
            self.path,
            train['image_id'].values, 
            train['label'].values,
            train_trans)
        # val dataset
        val_trans = torchvision.transforms.CenterCrop(self.size)
        self.val_dataset = Dataset(
            self.path,
            val['image_id'].values, 
            val['label'].values,
            val_trans)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)