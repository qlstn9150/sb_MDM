import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time

import tensorflow.compat.v1 as tf  # tensorflow 2.x
tf.disable_v2_behavior()  # tensorflow 2.x
tf.reset_default_graph()

from model import *

#callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=100)


def train(model_str, model_f, channel, epochs, L=8, k=4, n=4, batch_size=64):

    if channel == 'awgn':
        train_Eb_dB = 7
    elif channel == 'bursty':
        train_Eb_dB = 6
    else:
        train_Eb_dB = 12

    R = k / n
    noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))
    backendNoise = 1 / (2 * R * 10 ** (train_Eb_dB / 10))
    burst_beta = np.random.binomial(1, 0.05, size=(batch_size, L, 2 * n))

    _, vec_one_hot, label_one_hot = generate_data(200, k, L, batch_size)

    model = model_f(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    modelcheckpoint = ModelCheckpoint(filepath='./' + channel + '/' + model_str + '.h5',
                                      monitor='loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='auto', save_freq=1)
    start = time.clock()
    model.fit(vec_one_hot, label_one_hot,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_split=None, callbacks=[modelcheckpoint,reduce_lr,early_stopping])
    end = time.clock()
    print('The NN has trained ' + str(end - start) + ' s')


train(model_str = 'basic',  model_f = basic,
      channel='awgn', epochs=200)