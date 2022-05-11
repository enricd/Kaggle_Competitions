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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        img = np.concatenate((rgb[...,:2], np.expand_dims(nir, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:]), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        land = patch + '/' + str(observation_id) + '_landcover.tif'
        land = imread(land)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(land, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        alt = patch + '/' + str(observation_id) + '_altitude.tif'
        alt = imread(alt)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(alt, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        land = patch + '/' + str(observation_id) + '_landcover.tif'
        land = imread(land)
        alt = patch + '/' + str(observation_id) + '_altitude.tif'
        alt = imread(alt)
        img = np.concatenate((np.expand_dims(nir, axis=-1), rgb[...,-2:], np.expand_dims(land, axis=-1), np.expand_dims(alt, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        img = np.concatenate((rgb, np.expand_dims(nir, axis=-1)), axis=2)
        if self.trans is not None:
            img = self.trans(image=img)['image']
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
        rgb = patch + '/' + str(observation_id) + '_rgb.jpg'
        rgb = imread(rgb)
        nir = patch + '/' + str(observation_id) + '_near_ir.jpg'
        nir = imread(nir)
        bio = self.bio[ix].astype(np.float32)
        if self.trans is not None: # TODO: apply same transform to all images
            img = self.trans(image=img)['image']
        if self.labels is not None:
            label = self.labels[ix]
            return {'rgb': rgb, 'nir': nir, 'bio': bio,'label': label}
        return {'rgb': rgb, 'nir': nir, 'bio': bio,'observation_id': observation_id}  
    
    
