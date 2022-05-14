import torch
from skimage.io import imread
from .utils import get_patch, get_country
import numpy as np

class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ix):
        img = imread(self.images[ix])
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        
        observation_id = self.images[ix].split("/")[-1].split("_")[0]
        return img, observation_id
    
    
class RGNirDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        img = np.concatenate((rgb[...,:2], np.expand_dims(nir, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    
        
class NirGBDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:]), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    

class NirGBLandDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        land = patch + "/" + str(observation_id) + "_landcover.tif"
        land = imread(land)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(land, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    

class NirGBAltDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        alt = patch + "/" + str(observation_id) + "_altitude.tif"
        alt = imread(alt)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(alt, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    

class NirGBLandAltDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        land = patch + "/" + str(observation_id) + "_landcover.tif"
        land = imread(land)
        alt = patch + "/" + str(observation_id) + "_altitude.tif"
        alt = imread(alt)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(land, axis=-1), np.expand_dims(alt, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    

class RGBNirDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        img = np.concatenate((rgb, np.expand_dims(nir, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        return img, observation_id
    
    
class RGBNirBioDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, bio, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.bio = bio
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        bio = self.bio[ix].astype(np.float32)
        if self.trans is not None: # TODO: apply same transform to all images
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return {"rgb": rgb, "nir": nir, "bio": bio,"label": label}
        return {"rgb": rgb, "nir": nir, "bio": bio,"observation_id": observation_id}  
        
    

class NirGBAltBioDataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, bio, lat, lon, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.bio = bio
        self.lat = lat
        self.lon = lon
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        gb = rgb[...,-2:]
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        alt = patch + "/" + str(observation_id) + "_altitude.tif"
        alt = imread(alt)
        img = np.concatenate((np.expand_dims(nir, axis=-1), gb, np.expand_dims(alt, axis=-1)), axis=2)
        bio = self.bio[ix].astype(np.float32)
        country = get_country(observation_id)
        lat, lon = self.lat[ix], self.lon[ix]
        if self.trans is not None: 
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return {"img": img, "bio": bio, "country": country, "lat": lat, "lon": lon, "label": label}
        return {"img": img, "bio": bio, "country": country, "lat": lat, "lon": lon, "observation_id": observation_id}  
    
class NirGB_BioCAL_Dataset(torch.utils.data.Dataset):
    def __init__(self, observation_ids, bio, lat, lon, labels=None, trans=None):
        self.observation_ids = observation_ids
        self.bio = bio
        self.lat = lat
        self.lon = lon
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, ix):
        observation_id = self.observation_ids[ix]
        patch = get_patch(observation_id)
        # IMG data
        # GB
        rgb = patch + "/" + str(observation_id) + "_rgb.jpg"
        rgb = imread(rgb)
        gb = rgb[...,-2:]
        # Nir
        nir = patch + "/" + str(observation_id) + "_near_ir.jpg"
        nir = imread(nir)
        img = np.concatenate((np.expand_dims(nir, axis=-1), gb), axis=2)
        # MLP data (67 values total )
        # landcover --> "dot"hot encoder (0. if class not present, decimal percentage of amount if present), 34 values
        land = patch + "/" + str(observation_id) + "_landcover.tif"
        land = imread(land)
        f_dothot = lambda v: (img == v).sum() / 65536
        land_dothot = np.zeros(34)
        land_dothot[np.unique(land)] += [f_dothot(v) for v in np.unique(land)]
        # altitude --> min, max, dif
        alt = patch + "/" + str(observation_id) + "_altitude.tif"
        alt = imread(alt)
        max_alt, min_alt = alt.max() / 5000, alt.min() / 5000       # min-max normalization max altitude in France is about 4000m and in USA i about 6000m
        dif_alt = max_alt - min_alt
        # bio (environmental vectors), 27 values
        bio = self.bio[ix].astype(np.float32)
        # county byte, lat, lon
        country = get_country(observation_id)
        lat, lon = self.lat[ix], self.lon[ix]
        if self.trans is not None: 
            img = self.trans(image=img)["image"]
        if self.labels is not None:
            label = self.labels[ix]
            return {"img": img, 
                    "bio": bio, 
                    "country": country, 
                    "lat": lat, "lon": lon, 
                    "land": land_dothot.astype(np.float32),
                    "alt": np.array([max_alt, min_alt, dif_alt]).astype(np.float32),
                    "label": label}
        return {"img": img, 
                "bio": bio, 
                "country": country, 
                "lat": lat, "lon": lon, 
                "land": land_dothot.astype(np.float32),
                "alt": np.array([max_alt, min_alt, dif_alt]).astype(np.float32),
                "observation_id": observation_id}  
    

