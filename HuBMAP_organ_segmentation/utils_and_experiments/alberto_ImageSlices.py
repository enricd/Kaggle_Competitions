import cv2
import numpy as np

## ImageMaskSlices make slices of shape (Imsize, Imsize) of the Im and returns the upper-left coordinates of
## the slices which has labels on it and have less than a 30% of white background


def ImageMaskSlices(Im, Imsize,row, Imresize = 0):
    ImG = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    (thresh, Imbw) = cv2.threshold(ImG, 150, 1, cv2.THRESH_BINARY_INV)
    Imbw = BwRoi(Imbw)
    (OriImsize, _) = Imbw.shape
    Xslices = np.arange(0, OriImsize, Imsize)
    Xslices[-1] = OriImsize - Imsize
    Yslices = np.arange(0, OriImsize, Imsize)
    Yslices[-1] = OriImsize - Imsize

    Slices = []

    w = row['img_width']
    h = row['img_height']
    rle = row['rle']
    s = rle.split()
    starts, lengths = [np.asarray(t, dtype='int') for t in (s[0:][::2], s[1:][::2])]
    starts = starts - 1
    mask = np.zeros(h * w, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        mask[s:s + l] = 1
    mask = mask.reshape((h, w)).T
    if (Imresize > 0):
        mask = cv2.resize(mask, (Imresize, Imresize))


    for x in Xslices:
        for y in Yslices:
            MaskAny = mask[x:x+Imsize, y:y+Imsize].sum() != 0
            BwProp = Imbw[x:x+Imsize, y:y+Imsize].sum()/(Imsize**2)
            if((MaskAny == True) & (BwProp > 0.7)):
                Slices.append([x,y])
    return Slices


## BwRoi Dilates the part of the image with information to distinguish better from the white background

def BwRoi(Im):
    kernel = np.ones((15, 15), np.uint8)
    Imbw = cv2.dilate(Im, kernel, iterations=3)
    return Imbw

## ImageSlices takes slices trying to avoid the white background of the image

def ImageSlices(Im, Imsize):
    ImG = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    (thresh, Imbw) = cv2.threshold(ImG, 150, 1, cv2.THRESH_BINARY_INV)
    Imbw = BwRoi(Imbw)
    (s, _) = Imbw.shape
    samplepoint = s//2
    Npbw = np.array(Imbw)
    ecuador = np.where(Npbw[:, samplepoint] == 1)[0]
    maxw = ecuador[-1]-ecuador[0]
    Upper = True
    coors = []
    for col in np.arange(s):
        lin = np.where(Npbw[:, col] == 1)[0]
        if(lin.__len__() > 0):
            w = lin[-1] - lin[0]
        else:
            w = 0
        if(w>maxw/2):
            colend = np.where(Npbw[lin[0], :] == 1)[0][-1]
            coors.append([lin[0],col])
            coors.append([lin[-1], colend])
            break

    ImG[coors[0][0]:coors[1][0], coors[0][1]: coors[1][1]] = 0


    wTrim = coors[1][0] - coors[0][0]
    hTrim = coors[1][1] - coors[0][1]

    wSlices = int(np.ceil(wTrim/Imsize))
    hSlices = int(np.ceil(hTrim /Imsize))

    ImSlices = np.zeros((wSlices*hSlices, Imsize, Imsize,3), dtype=int)

    Xslices = np.arange(coors[0][0], coors[1][0], Imsize)
    Xslices[-1] = coors[1][0] - Imsize
    Yslices = np.arange(coors[0][1], coors[1][1], Imsize)
    Yslices[-1] = coors[1][1] - Imsize

    i = 0
    for x in Xslices:
        for y in Yslices:
            ImSlices[i] = Im[x:x+Imsize, y:y+Imsize]
            i+=1

    return Imbw, ImSlices, Xslices, Yslices

