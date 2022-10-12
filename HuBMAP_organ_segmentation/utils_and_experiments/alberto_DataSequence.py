import segmentation_models as sm
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from segmentation_models.utils import set_trainable

sm.set_framework('tf.keras')
sm.framework()

## SliceSequence resizes each image to img_Resize and takes slices as stored by the script Dfslices in the Dataframe

class SliceSequence(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, df, img_Resize = 0, shuffle = True):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.df = df
        self.img_Resize = img_Resize
        self.indexes = np.arange(0, int(np.floor(len(df)/batch_size))*batch_size)
        if(shuffle):
            np.random.shuffle(self.indexes)


    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def __getitem__(self, idx):
        ind = self.indexes[idx:idx+self.batch_size]

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype='uint8')
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype='uint8')

        IdPrev = -1
        for j, (indexm, row) in enumerate(self.df.iloc[ind].iterrows()):
            if(IdPrev != int(row['id'])):
                img_path = r"./data/train_images/" + str(int(row['id'])) + ".tiff"
                if(self.img_Resize > 0):
                    img = tf.keras.utils.load_img(img_path, target_size=(self.img_Resize, self.img_Resize))
                else:
                    img = tf.keras.utils.load_img(img_path)
                img_array = tf.keras.utils.img_to_array(img, dtype='uint8')
            Xslice = int(row['Xslice'])
            Yslice = int(row['Yslice'])
            x[j] = img_array[Xslice:Xslice+self.img_size, Yslice:Yslice+self.img_size]

            # calculate mask
            if (IdPrev != int(row['id'])):
                w = int(row['img_width'])
                h = int(row['img_height'])
                rle = row['rle']
                s = rle.split()
                starts, lengths = [np.asarray(t, dtype='int') for t in (s[0:][::2], s[1:][::2])]
                starts = starts - 1
                original_mask = np.zeros(h * w, dtype=np.uint8)
                for s, l in zip(starts, lengths):
                    original_mask[s:s + l] = 1
                original_mask = original_mask.reshape((h, w)).T
                original_mask = tf.keras.utils.array_to_img(original_mask[:, :, tf.newaxis], scale=False)

                # resize mask
                if (self.img_Resize > 0):
                    original_mask = original_mask.resize((self.img_Resize, self.img_Resize))
                mask_array = tf.keras.utils.img_to_array(original_mask, dtype='uint8')
            y[j] = mask_array[Xslice:Xslice+self.img_size, Yslice:Yslice+self.img_size]
            IdPrev = int(row['id'])
        return x.astype('float32') / 255, y

## ImSequence with a resize of each image to img_size

class ImSequence(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, df):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.df = df


    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def __getitem__(self, idx):
        ind = np.arange(idx * self.batch_size, (idx+1)*self.batch_size)%len(self.df)

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype='uint8')
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype='uint8')

        for j, (indexm, row) in enumerate(self.df.iloc[ind].iterrows()):
            img_path = r"./data/train_images/" + str(row['id']) + ".tiff"
            img = tf.keras.utils.load_img(img_path,
                                       target_size=(self.img_size, self.img_size))
            img_array = tf.keras.utils.img_to_array(img, dtype='uint8')
            x[j] = img_array

            # calculate mask
            w = row['img_width']
            h = row['img_height']
            rle = row['rle']
            s = rle.split()
            starts, lengths = [np.asarray(t, dtype='int') for t in (s[0:][::2], s[1:][::2])]
            starts = starts - 1
            original_mask = np.zeros(h * w, dtype=np.uint8)
            for s, l in zip(starts, lengths):
                original_mask[s:s + l] = 1
            original_mask = original_mask.reshape((h, w)).T
            # we need at least three channels to save an image so here expand mask to (3000, 3000, 1)
            original_mask = tf.keras.utils.array_to_img(original_mask[:, :, tf.newaxis], scale=False)

            # resize mask
            mask = original_mask.resize((self.img_size, self.img_size))
            mask_array = tf.keras.utils.img_to_array(mask, dtype='uint8')
            y[j] = mask_array.reshape((self.img_size,self.img_size,1))
        return x.astype('float32') / 255, y