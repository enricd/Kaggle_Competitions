import segmentation_models as sm
from segmentation_models import Unet
import keras
from keras import utils as np_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alberto_DataSequence import SliceSequence, ImSequence


sm.set_framework('tf.keras')
sm.framework()


model = Unet(backbone_name='efficientnetb7', encoder_weights='imagenet', encoder_freeze=True)
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])


batch_size = 4
# size of the input square image for the model
img_size = 256
# Parameter used for an intermediate resize for SliceSequence
img_Resize = 1024

df = pd.read_csv('./data/train.csv')

# Filter the data related to the organ with which is going to be tested

#df = df[df.organ == 'spleen']

# Number of samples taken for validation

NVal = int(np.floor(len(df)*0.2))

dfshuffle = df.iloc[np.random.permutation(len(df))]

dv = dfshuffle.iloc[-NVal:]
dt = dfshuffle.iloc[:-NVal]

## ImSequence with a resize of each image to img_size

Train = ImSequence(batch_size, img_size, dt)
Val = ImSequence(batch_size, img_size, dv)


## SliceSequence resizes each image to img_Resize and takes slices as stored by the script Dfslices in the Dataframe

#Train = SliceSequence(batch_size, img_size, dt, img_Resize=img_Resize,  shuffle=False)
#Val = SliceSequence(batch_size, img_size, dv, img_Resize=img_Resize, shuffle=False)


call = None#keras.callbacks.ModelCheckpoint('efficient_unet_model', save_best_only=True)

history = model.fit(Train, epochs=15, validation_data = Val)

epochs = range(len(history.history['loss']))
plt.subplot(1,2,1)
plt.plot(epochs, history.history['loss'], label='train loss')
plt.subplot(1,2,2)
plt.plot(epochs, history.history['val_loss'], label='val loss')
plt.legend()
plt.show()