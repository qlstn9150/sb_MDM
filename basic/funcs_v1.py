from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, SimpleRNN, LSTM, \
    GRU, Bidirectional, TimeDistributed, Activation, Concatenate, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as KR
import numpy as np


'''
 --- COMMUNICATION PARAMETERS ---
'''


# Bits per Symbol
k = 4

# Number of symbols
L = 8

# Channel Use
n = 4


# Effective Throughput
#  bits per symbol / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 7

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))

# Number of messages used for test, each size = k*L
batch_size = 64
nb_train_word = batch_size * 200
num_of_sym = batch_size * 1000


# Define Power Norm for Tx
def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels


# Define Channel Layers including AWGN and Flat Rayleigh fading
#  x: input data
#  sigma: noise std
def channel_layer(x, sigma):
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)

    return x + w


# %%
def model_dnn(noise_sigma):
    num_fiters_1 = 1
    kernel_size_1 = 256

    model = Sequential([
        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('linear'),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),

        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        # Define Decoder Layers (Receiver)
        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        # Output One hot vector and use Softmax to soft decoding
        Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax'),
    ])
    return model


# %%
def model_cnn(noise_sigma):
    num_fiters_1 = 256
    kernel_size_1 = 1

    model = Sequential([
        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('linear'),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),

        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        # Define Decoder Layers (Receiver)
        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        # Output One hot vector and use Softmax to soft decoding
        Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax'),
    ])
    return model


def model_densenet(noise_sigma):
    num_fiters_1 = 64
    kernel_size_1 = 1
    kernel_size_2 = 3

    input = Input(batch_shape=(batch_size, L, 2 ** k))
    e1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(input)
    e1 = BatchNormalization()(e1)
    e1 = Activation('elu')(e1)

    e2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = Activation('elu')(e2)

    e2 = Concatenate()([e1, e2])

    e3 = Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('linear')(e3)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e3)

    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    # Output One hot vector and use Softmax to soft decoding
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)

    return model


def model_resnet(noise_sigma):
    num_fiters_1 = 64
    kernel_size_1 = 1
    kernel_size_2 = 3

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(input)
    e1 = BatchNormalization()(e1)
    e1 = Activation('elu')(e1)

    e2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = Activation('elu')(e2)

    e2 = Add()([e1, e2])

    e3 = Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('linear')(e3)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e3)

    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Add()([d1, d2])

    # Output One hot vector and use Softmax to soft decoding
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)

    return model


# %%
def model_cnn_gru(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 1
    num_nodes_1 = 128

    model = Sequential([

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        # LSTM(num_nodes_1, return_sequences=True),
        # GRU(num_nodes_1, return_sequences=True),
        # TimeDistributed(Dense(2*n, )),

        # LSTM(2*n, return_sequences=True),
        GRU(2 * n, return_sequences=True),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),

        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        # LSTM(num_nodes_1, return_sequences=True),
        GRU(num_nodes_1, return_sequences=True),

        # TimeDistributed(Dense(units=num_nodes_1,)),

        Conv1D(filters=2 ** k, strides=1, kernel_size=1, padding='same', activation='softmax')
    ])

    return model

'''
def model_gru(noise_sigma):

    num_nodes_1 = 64
    num_nodes_2 = 128

    model = Sequential([

        GRU(num_nodes_1, return_sequences=True, input_shape=(L, 2 ** k)),
        # LSTM(num_nodes_1, return_sequences=True),
        # GRU(num_nodes_1, return_sequences=True),
        # TimeDistributed(Dense(2*n, )),

        # LSTM(2*n, return_sequences=True),
        Dense(2 * n),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),

        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        # LSTM(num_nodes_1, return_sequences=True),
        GRU(num_nodes_1, return_sequences=True),

        # TimeDistributed(Dense(units=num_nodes_1,)),

        Dense(2 ** k, activation='softmax')
    ])

    return model
'''


def model_gru(noise_sigma):

    num_fiters_1 = 128
    kernel_size_1 = 1
    num_nodes_1 = 128
    num_nodes_2 = 128

    model = Sequential([

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        Dense(2 * n),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),

        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        # LSTM(num_nodes_1, return_sequences=True),
        GRU(num_nodes_1, return_sequences=True),

        # TimeDistributed(Dense(units=num_nodes_1,)),

        Dense(2 ** k, activation='softmax')
    ])

    return model

def model13():
    R = k / n
    noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))
    num_fiters_1 = 128
    kernel_size_1 = 1
    num_nodes_1 = 128

    model = Sequential([

        Conv1D(128, 1, strides=1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        SimpleRNN(2*n, return_sequences=True),
        #GRU(2 * n, return_sequences=True),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),
        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        SimpleRNN(128, return_sequences=True),

        Conv1D(filters=2 ** k, strides=1, kernel_size=1, padding='same', activation='softmax')
    ])
    return model