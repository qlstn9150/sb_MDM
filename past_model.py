from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, BatchNormalization
from tensorflow.keras.layers import Input, Conv1D, SimpleRNN, LSTM
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed, Activation
from tensorflow.keras.layers import Concatenate, Add, Embedding, Reshape
from tensorflow.keras.layers import MaxPooling1D, Cropping1D, SimpleRNN, TimeDistributed
from tensorflow.keras.layers import PReLU, Conv1DTranspose, UpSampling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as KR
import numpy as np

k = 4
L = 8
n = 4
R = k / n
train_Eb_dB = 7
batch_size = 64
nb_train_word = batch_size * 200
num_of_sym = batch_size * 1000
noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))

def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels

def awgn_channel_layer(x, sigma):
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)
    return x + w

def bursty_channel_layer(x):
    backendNoise = 1 / (2 * R * 10 ** (train_Eb_dB / 10))  # Noise Variance
    alpha = 0.05  # Probability of burst noise
    burst_beta = np.random.binomial(1, alpha, size=(batch_size, L, 2 * n))
    burstyNoise = 1.0  # Set the bursty noise variance

    n1 = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(burstyNoise))
    n2 = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(backendNoise))
    return x + burst_beta*n1 + n2

def complex_multi(h,x):
    tmp_array = KR.ones(shape=(KR.shape(x)[0], L, 1))
    n_sign_array = KR.concatenate([tmp_array, -tmp_array], axis=2)
    h1 = h * n_sign_array
    h2 = KR.reverse(h, axes=2)
    tmp = h1 * x
    h1x = KR.sum(tmp, axis=-1)
    tmp = h2 * x
    h2x = KR.sum(tmp, axis=-1)
    a_real = KR.expand_dims(h1x, axis=2)
    a_img = KR.expand_dims(h2x, axis=2)
    a_complex_array = KR.concatenate([a_real, a_img], axis=-1)
    return a_complex_array

def rayleigh_channel_layer(x, sigma):
    a_complex = []
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)
    h = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(1 / 2))
    for i in range(0,2*n,2):
        y_h = complex_multi(h[:,:,i:i+2],x[:,:,i:i+2])
        if i ==0:
            a_complex = y_h
        else:
            a_complex = KR.concatenate([a_complex,y_h],axis=-1)
    result = KR.concatenate([a_complex+w,h],axis=-1)
    return result


# MAKE MODELS
def basic(noise_sigma):
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

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)
    return model

def model5(noise_sigma):
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

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d2, d3])

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model6(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32

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

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d2, d3])

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model7(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32

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

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d1, d2, d3])

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model8(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    #e1 = Dense(2 * n)(input)

    e1 = GRU(2 * n, return_sequences=True)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d1, d2, d3])

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model9(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels

    e1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_3, padding='same')(input)
    e1 = BatchNormalization()(e1)
    e1 = Activation('elu')(e1)

    e2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = Activation('elu')(e2)

    e2 = Add()([e1, e2])

    e3 = Conv1D(filters=2*n, strides=1, kernel_size=kernel_size_1, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('elu')(e3)

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

    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)
    return model

def model10(noise_sigma):
    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = GRU(128, return_sequences=True)(c)

    d1 = Conv1D(128, 32, strides=1, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(64, 4, strides=1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d3 = Conv1D(32, 4, strides=1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    output = Conv1D(2 ** k , 1, strides=1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model11(noise_sigma):
    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = LSTM(128, return_sequences=True)(c)

    d1 = Conv1D(128, 32, strides=1, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(64, 32, strides=1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d3 = Conv1D(32, 32, strides=1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    output = Conv1D(2 ** k , 1, strides=1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

def model12(noise_sigma):
    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    e4 = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e4)

    # Define Decoder Layers (Receiver)
    d1 = LSTM(128, return_sequences=True)(c)

    d1 = Conv1D(128, 1, strides=1, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(64, 1, strides=1, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d3 = Conv1D(32, 1, strides=1, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    output = Conv1D(2 ** k , 1, strides=1, padding='same', activation='softmax')(d3)

    model = Model(inputs=input, outputs=output)
    return model

#1102
def model13(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    c = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(awgn_channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(c)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Add()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Add()([d1, d2, d3])

    d4 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('elu')(d4)

    d4 = Add()([d1, d2, d3, d4])

    # Output One hot vector and use Softmax to soft decoding
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d4)

    model = Model(inputs=input, outputs=output)

    return model

def model14(noise_sigma):
    num_fiters_1 = 128
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    c = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    c = Lambda(awgn_channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(c)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(c)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Concatenate()([d1, d2])

    d3 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('elu')(d3)

    d3 = Concatenate()([d1, d2, d3])

    d4 = Conv1D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('elu')(d4)

    d4 = Concatenate()([d1, d2, d3, d4])

    # Output One hot vector and use Softmax to soft decoding
    output = Conv1D(filters=2 ** k, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d4)

    model = Model(inputs=input, outputs=output)
    return model