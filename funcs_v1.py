from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, BatchNormalization
from tensorflow.keras.layers import Input, Conv1D, SimpleRNN, LSTM
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed, Activation
from tensorflow.keras.layers import Concatenate, Add, Embedding, Reshape
from tensorflow.keras.layers import MaxPooling1D, Cropping1D, SimpleRNN, TimeDistributed
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
batch_size = 64 #364nb_train_word = batch_size * 200
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

def basic(noise_sigma):
    num_fiters_1 = 256
    kernel_size_1 = 1

    model = Sequential()
    # Define Encoder Layers
    model.add(Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', name='e_1',
                     input_shape=(L, 2 ** k)))
    model.add(BatchNormalization(name='e_2'))
    model.add(Activation('elu', name='e_3'))

    model.add(Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', name='e_7'))
    model.add(BatchNormalization(name='e_8'))
    model.add(Activation('elu', name='e_9'))

    model.add(Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same', name='e_10'))
    model.add(BatchNormalization(name='e_11'))
    model.add(Activation('linear', name='e_12'))

    # Tx Power normalization
    model.add(Lambda(normalization, name='power_norm'))
    # AWGN channel
    model.add(Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'))

    # Define Decoder Layers (Receiver)
    model.add(Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', name='d_1'))
    model.add(BatchNormalization(name='d_2'))
    model.add(Activation('elu', name='d_3'))

    model.add(Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', name='d_7'))
    model.add(BatchNormalization(name='d_8'))
    model.add(Activation('elu', name='d_9'))

    # Output One hot vector and use Softmax to soft decoding
    model.add(
        Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', name='d_10', activation='softmax'))

    return model


###basic
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

###best
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

def model_cnn_gru(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 1
    num_nodes_1 = 128

    model = Sequential([

        Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        GRU(2 * n, return_sequences=True),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),
        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        GRU(num_nodes_1, return_sequences=True),

        Conv1D(filters=2 ** k, strides=1, kernel_size=1, padding='same', activation='softmax')
    ])

    return model

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

def model9(noise_sigma):
    num_fiters_1 = 64
    num_fiters_2 = 128
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

    e3 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('elu')(e3)

    e4 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_1, padding='same')(e3)
    e4 = BatchNormalization()(e4)
    e4 = Activation('elu')(e4)

    e4 = Concatenate()([e3, e4])

    e5 = Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same')(e4)
    e5 = BatchNormalization()(e5)
    en_output = Activation('linear')(e5)

    ########################################################
    # Tx Power normalization
    x = Lambda(normalization, name='power_norm')(en_output)
    # AWGN channel
    ch_output = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(x)
    #########################################################

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(ch_output)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d4 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_1, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('elu')(d4)

    d4 = Concatenate()([d3, d4])

    # Output One hot vector and use Softmax to soft decoding
    de_output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d4)

    model = Model(inputs=input, outputs=de_output)

    return model

def model10(noise_sigma):
    num_fiters_1 = 64
    num_fiters_2 = 128
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

    e3 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('elu')(e3)

    e4 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_1, padding='same')(e3)
    e4 = BatchNormalization()(e4)
    e4 = Activation('elu')(e4)

    e4 = Concatenate()([e3, e4])

    e5 = Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_1, padding='same')(e4)
    e5 = BatchNormalization()(e5)
    en_output = Activation('linear')(e5)

    ########################################################
    # Tx Power normalization
    x = Lambda(normalization, name='power_norm')(en_output)
    # AWGN channel
    ch_output = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(x)
    #########################################################

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_1, padding='same')(ch_output)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d4 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('elu')(d4)

    d4 = Concatenate()([d3, d4])

    # Output One hot vector and use Softmax to soft decoding
    de_output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d4)

    model = Model(inputs=input, outputs=de_output)

    return model

def model11(noise_sigma):
    num_fiters_1 = 64
    num_fiters_2 = 128
    kernel_size_1 = 1
    kernel_size_2 = 3

    input = Input(batch_shape=(batch_size, L, 2 ** k))
    e1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(input)
    e1 = BatchNormalization()(e1)
    e1 = Activation('elu')(e1)

    e2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = Activation('elu')(e2)

    e2 = Concatenate()([e1, e2])

    e3 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('elu')(e3)

    e3 = Concatenate()([e2, e3])

    e4 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(e3)
    e4 = BatchNormalization()(e4)
    e4 = Activation('elu')(e4)

    e4 = Concatenate()([e3, e4])

    e5 = Conv1D(filters=2 * n, strides=1, kernel_size=kernel_size_2, padding='same')(e4)
    e5 = BatchNormalization()(e5)
    en_output = Activation('linear')(e5)

    ########################################################
    # Tx Power normalization
    x = Lambda(normalization, name='power_norm')(en_output)
    # AWGN channel
    ch_output = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(x)
    #########################################################

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(ch_output)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d2, d3])

    d4 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('elu')(d4)

    d4 = Concatenate()([d3, d4])

    # Output One hot vector and use Softmax to soft decoding
    de_output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d4)

    model = Model(inputs=input, outputs=de_output)

    return model

def model12(noise_sigma):
    num_nodes_1 = 128

    model = Sequential([

        Conv1D(128, 1, strides=1, padding='same', input_shape=(L, 2 ** k)),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(128, 3, strides=1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        LSTM(2*n, return_sequences=True),
        #GRU(2 * n, return_sequences=True),

        # Tx Power normalization
        Lambda(normalization, name='power_norm'),
        # AWGN channel
        Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer'),

        LSTM(num_nodes_1, return_sequences=True),

        Conv1D(128, 3, strides=1, padding='same'),
        BatchNormalization(),
        Activation('elu'),

        Conv1D(filters=2 ** k, strides=1, kernel_size=1, padding='same', activation='softmax')
    ])
    return model

def model13(noise_sigma):
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

####best
def model1(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)

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
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)

    return model

#parameter
def model2(noise_sigma):
    num_fiters_1 = 64
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)

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
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)

    return model

def model3(noise_sigma):
    num_fiters_1 = 64
    num_fiters_2 = 16

    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)

    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_2, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    # Output One hot vector and use Softmax to soft decoding
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)

    return model
