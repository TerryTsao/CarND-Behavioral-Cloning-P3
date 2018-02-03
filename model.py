import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle


# Constants
DATA_DIR = 'data_sdc'
IMG_DIR = '{}/IMG'.format(DATA_DIR)
correction = .35



"""
Process each row and return image and steering data
"""
def process_batch_helper(row):
    img_filename = '{}/{}'.format(IMG_DIR, row.img_filename)
    image = cv2.imread(img_filename)
    if row.flip == -1:
        image = np.fliplr(image)
    steering = row.steering * row.flip
    return image, steering



"""
Generator of each batch
"""
def batch_generator(df, batch_size=32):
    while True:
        shuffle(df)
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size, :]

            images = []
            steerings = []
            for _, row in batch_df.iterrows():
                image, steering = process_batch_helper(row)
                images.append(image)
                steerings.append(steering)

            yield shuffle(np.array(images), np.array(steerings))



# get data info
from log_file_preprocess import preprocess
sample_df = preprocess(correction)

# train validation split
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(sample_df, test_size=.2)

train_gen = batch_generator(train_df)
valid_gen = batch_generator(valid_df)


# CNN
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D,\
        MaxPooling2D, Dropout, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - .5, input_shape=(160,320,3)))
model.add(Cropping2D(((70,25), (0,0))))   # 65, 320, 3
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), activation='relu', ))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# uncomment to continue learning
#model = load_model('./model.h5')

model.fit_generator(train_gen, steps_per_epoch=len(train_df),
        validation_data=valid_gen, validation_steps=len(valid_df),
        epochs=1, )

model.save('model.h5')
