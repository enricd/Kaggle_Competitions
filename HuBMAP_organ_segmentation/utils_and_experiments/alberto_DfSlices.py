import pandas as pd
import numpy as np
import cv2
from alberto_ImageSlices import ImageSlices
from alberto_ImageSlices import ImageMaskSlices

ImageResize = 1024

ImageSize = 256
d = pd.read_csv('./data/train.csv')

ds = d[d.organ == 'spleen']


df = pd.DataFrame()
df['Xslice'] = int(0)
df['Yslice'] = int(0)
df['img_width'] = int(0)
df['img_height'] = int(0)
df['id'] = int(0)
df['rle'] = ""
s = 0

for i, row in ds.iterrows():
    path = r"./data/train_images/" + str(row['id']) + ".tiff"
    im = cv2.imread(path)
    print(row['id'])
    im = cv2.resize(im, (ImageResize, ImageResize))
    #_, _, xs, ys = ImageSlices(im, 256)
    '''for x in xs:
        for y in ys:
            df.loc[s, 'Xslice'] = int(x)
            df.loc[s, 'Yslice'] = int(y)
            df.loc[s, 'img_width']  = int(ds.loc[i, 'img_width'])
            df.loc[s, 'img_height'] = int(ds.loc[i, 'img_height'])
            df.loc[s, 'id'] = int(ds.loc[i, 'id'])
            df.loc[s, 'rle'] = ds.loc[i, 'rle']
            s+=1'''
    slices = ImageMaskSlices(im, 256, row, ImageResize)
    for slice in slices:
        df.loc[s, 'Xslice'] = int(slice[0])
        df.loc[s, 'Yslice'] = int(slice[1])
        df.loc[s, 'img_width'] = int(ds.loc[i, 'img_width'])
        df.loc[s, 'img_height'] = int(ds.loc[i, 'img_height'])
        df.loc[s, 'id'] = int(ds.loc[i, 'id'])
        df.loc[s, 'rle'] = ds.loc[i, 'rle']
        s += 1

df.to_csv('TrainSliced4.csv')


