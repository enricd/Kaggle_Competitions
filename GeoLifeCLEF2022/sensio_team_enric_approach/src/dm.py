import numpy as np
import pytorch_lightning as pl
import os
from pathlib import Path
import pandas as pd
from .utils import get_patch_rgb
from .ds import RGBDataset, RGBNirDataset, NirGBDataset, RGNirDataset, NirGBLandDataset, NirGBAltDataset, NirGBLandAltDataset, RGBNirBioDataset, NirGBAltBioDataset, NirGB_BioCAL_Dataset, RGBNir_BioCAL_Dataset, RGBNir_BioCAL_MultiLabel_Dataset
from torch.utils.data import DataLoader
import albumentations as A
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json

class RGBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path = "../data", num_workers=0, pin_memory=False, train_trans=None, test_trans=None, freq_subset=None):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.test_trans = test_trans
        self.freq_subset = freq_subset
        
    def read_data(self, mode="train"):
        obs_fr = pd.read_csv(self.path / "observations" / f"observations_fr_{mode}.csv", sep=";")
        obs_us = pd.read_csv(self.path / "observations" / f"observations_us_{mode}.csv", sep=";")
        obs = pd.concat([obs_fr, obs_us])
        if mode == "train":
            multi_labels = pd.read_csv("obs_multi_labels.csv", index_col="observation_id")
            obs["multi_labels"] = list(map(lambda x: json.loads(x[0]), multi_labels.values))
        if self.freq_subset is not None and mode == "train":
            species_value_counts = obs["species_id"].value_counts()
            split = 600
            freq_subset_species_ids = species_value_counts[:split].index if self.freq_subset == "high" else species_value_counts[split:].index
            obs = obs[obs["species_id"].isin(freq_subset_species_ids)]
        
        return obs

    def split_data(self):
        #self.data_train = self.data[self.data["subset"] == "val"]
        self.data_train = self.data
        self.data_val = self.data[self.data["subset"] == "val"]

    def generate_datasets(self):
        self.ds_train = RGBDataset(
            self.data_train.image.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBDataset(
            self.data_val.image.values, self.data_val.species_id.values)
        self.ds_test = RGBDataset(self.data_test.image.values)

    def print_dataset_info(self):
        print("train:", len(self.ds_train))
        print("val:", len(self.ds_val))
        print("test:", len(self.ds_test))
    
    def setup(self, stage=None):
        self.data = self.read_data()
        self.data["image"] = self.data["observation_id"].apply(get_patch_rgb)
        self.data_test = self.read_data("test")
        self.data_test["image"] = self.data_test["observation_id"].apply(get_patch_rgb)
        self.split_data()
        self.generate_datasets()
        self.print_dataset_info()
        
        print("train:", len(self.ds_train))
        print("val:", len(self.ds_val))
        print("test:", len(self.ds_test))
        
    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size = batch_size if batch_size is not None else self.batch_size,
            shuffle = shuffle if shuffle is not None else True,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,)
    
    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)
        
    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)
        
    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)
    

class RGNirDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = RGNirDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGNirDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = RGNirDataset(self.data_test.observation_id.values)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        self.generate_datasets()
        self.print_dataset_info()
        

class NirGBDataModule(RGNirDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = NirGBDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = NirGBDataset(self.data_test.observation_id.values)
     
        
class NirGBLandDataModule(RGNirDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = NirGBLandDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBLandDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = NirGBLandDataset(self.data_test.observation_id.values)
    
    
class NirGBAltDataModule(RGNirDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = NirGBAltDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBAltDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = NirGBAltDataset(self.data_test.observation_id.values)
    
        
class NirGBLandAltDataModule(RGNirDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = NirGBLandAltDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBLandAltDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = NirGBLandAltDataset(self.data_test.observation_id.values)
        

class RGBNirDataModule(RGNirDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = RGBNirDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNirDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = RGBNirDataset(self.data_test.observation_id.values)
        
        
class RGBNirBioDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = RGBNirBioDataset(
            self.data_train.observation_id.values, self.X_train, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNirBioDataset(
            self.data_val.observation_id.values, self.X_val, self.data_val.species_id.values)
        self.ds_test = RGBNirBioDataset(self.data_test.observation_id.values, self.X_test)     
        

class NirGBAltBioDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None, test_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans, test_trans)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = NirGBAltBioDataset(
            self.data_train.observation_id.values, 
            self.X_train, 
            self.data_train.latitude.values, self.data_train.longitude.values, 
            self.data_train.species_id.values, 
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBAltBioDataset(
            self.data_val.observation_id.values,
            self.X_val,
            self.data_val.latitude.values, self.data_val.longitude.values, 
            self.data_val.species_id.values
            )
        self.ds_test = NirGBAltBioDataset(
            self.data_test.observation_id.values, 
            self.X_test,
            self.data_test.latitude.values, self.data_test.longitude.values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ]) if self.test_trans is not None else None
            )  
        

class NirGB_BioCAL_DataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None, test_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans, test_trans)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([("imputer", SimpleImputer(
                                missing_values=np.nan,
                                strategy="constant",
                                fill_value=np.finfo(np.float32).min,
                            )),
            ("std_scaler", StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = NirGB_BioCAL_Dataset(
            self.data_train.observation_id.values,  # observation_ids
            self.X_train,                           # bio
            self.data_train.latitude.values, self.data_train.longitude.values,  # lat, lon
            self.data_train.species_id.values,      # labels
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGB_BioCAL_Dataset(
            self.data_val.observation_id.values,
            self.X_val,
            self.data_val.latitude.values, self.data_val.longitude.values, 
            self.data_val.species_id.values
            )
        self.ds_test = NirGB_BioCAL_Dataset(
            self.data_test.observation_id.values, 
            self.X_test,
            self.data_test.latitude.values, self.data_test.longitude.values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ]) if self.test_trans is not None else None
            ) 

    
class RGBNir_BioCAL_DataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None, test_trans=None, freq_subset=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans, test_trans, freq_subset)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([("imputer", SimpleImputer(
                                missing_values=np.nan,
                                strategy="constant",
                                fill_value=np.finfo(np.float32).min,
                            )),
            ("std_scaler", StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = RGBNir_BioCAL_Dataset(
            self.data_train.observation_id.values,  # observation_ids
            self.X_train,                           # bio
            self.data_train.latitude.values, self.data_train.longitude.values,  # lat, lon
            self.data_train.species_id.values,      # labels
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNir_BioCAL_Dataset(
            self.data_val.observation_id.values,
            self.X_val,
            self.data_val.latitude.values, self.data_val.longitude.values, 
            self.data_val.species_id.values
            )
        self.ds_test = RGBNir_BioCAL_Dataset(
            self.data_test.observation_id.values, 
            self.X_test,
            self.data_test.latitude.values, self.data_test.longitude.values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ]) if self.test_trans is not None else None
            )


class RGBNir_BioCAL_MultiLabel_DataModule(RGBDataModule):
    def __init__(self, batch_size=32, path="../data", num_workers=0, pin_memory=False, train_trans=None, test_trans=None, freq_subset=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans, test_trans, freq_subset)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data("test")
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([("imputer", SimpleImputer(
                                missing_values=np.nan,
                                strategy="constant",
                                fill_value=np.finfo(np.float32).min,
                            )),
            ("std_scaler", StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = RGBNir_BioCAL_MultiLabel_Dataset(
            self.data_train.observation_id.values,  # observation_ids
            self.X_train,                           # bio
            self.data_train.latitude.values, self.data_train.longitude.values,  # lat, lon
            self.data_train.species_id.values,        # labels
            self.data_train.multi_labels.values,     
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNir_BioCAL_MultiLabel_Dataset(
            self.data_val.observation_id.values,
            self.X_val,
            self.data_val.latitude.values, self.data_val.longitude.values, 
            self.data_val.species_id.values,
            self.data_val.multi_labels.values,
            )
        self.ds_test = RGBNir_BioCAL_MultiLabel_Dataset(
            self.data_test.observation_id.values, 
            self.X_test,
            self.data_test.latitude.values, self.data_test.longitude.values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ]) if self.test_trans is not None else None
            )



