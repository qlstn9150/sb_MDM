from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, BatchNormalization
from tensorflow.keras.layers import Input, Conv1D, SimpleRNN, LSTM
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed, Activation
from tensorflow.keras.layers import Concatenate, Add, Embedding, Reshape
from tensorflow.keras.layers import MaxPooling1D, Cropping1D, SimpleRNN, TimeDistributed
from tensorflow.keras.layers import PReLU, Conv1DTranspose, UpSampling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as KR
import numpy as np
from tensorflow.keras.utils import to_categorical
import copy

import tensorflow.compat.v1 as tf  # tensorflow 2.x
tf.disable_v2_behavior()  # tensorflow 2.x
tf.reset_default_graph()



def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels

def generate_data(a, k, L, batch_size):
        nb_train_word = batch_size * a
        train_data = np.random.randint(low=0, high=2, size=(nb_train_word, k * L))
        label_data = copy.copy(train_data)
        train_data = np.reshape(train_data, newshape=(nb_train_word, L, k))
        tmp_array = np.zeros(shape=k)
        for i in range(k):
            tmp_array[i] = 2 ** i
        int_data = tmp_array[::-1]
        int_data = np.reshape(int_data, newshape=(k, 1))
        one_hot_data = np.dot(train_data, int_data)
        vec_one_hot = to_categorical(y=one_hot_data, num_classes=2 ** k)
        label_one_hot = copy.copy(vec_one_hot)
        return one_hot_data, vec_one_hot, label_one_hot

def awgn(x, sigma):
        w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)
        return x + w

def bursty(x, backendNoise, burst_beta):
        # Set the bursty noise variance : 1.0
        n1 = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(1.0))
        n2 = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(backendNoise))
        return x + burst_beta*n1 + n2

def rayleigh(x, L, n, sigma):
        def complex_multi(h, x):
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


# MAKING MODELS
def basic(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    model_input = Input(batch_shape=(None, L, 2 ** k), name='input_bits')

    e = Conv1D(filters=256, strides=1, kernel_size=1, name='e_1')(model_input)
    e = BatchNormalization(name='e_2')(e)
    e = Activation('elu', name='e_3')(e)

    e = Conv1D(filters=256, strides=1, kernel_size=1, name='e_7')(e)
    e = BatchNormalization(name='e_8')(e)
    e = Activation('elu', name='e_9')(e)

    e = Conv1D(filters=2 * n, strides=1, kernel_size=1, name='e_10')(e)  # 2 = I and Q channels
    e = BatchNormalization(name='e_11')(e)
    e = Activation('linear', name='e_12')(e)

    c = Lambda(normalization, name='power_norm')(e)

    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c, backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

    # Define Decoder Layers (Receiver)
    d = Conv1D(filters=256, strides=1, kernel_size=1, name='d_1')(c)
    d = BatchNormalization(name='d_2')(d)
    d = Activation('elu', name='d_3')(d)

    d = Conv1D(filters=256, strides=1, kernel_size=1, name='d_7')(d)
    d = BatchNormalization(name='d_8')(d)
    d = Activation('elu', name='d_9')(d)

    # Output One hot vector and use Softmax to soft decoding
    model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d)

    # Build System Model
    model = Model(model_input, model_output)
    return model

def model1(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    num_fiters_1 = 128
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    c = Lambda(normalization, name='power_norm')(e1)
    # AWGN channel
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c, backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

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

def model2(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    model_input = Input(batch_shape=(None, L, 2 ** k), name='input_bits')

    e1 = Conv1D(filters=256, strides=1, kernel_size=1, name='e_1')(model_input)
    e1 = BatchNormalization(name='e_2')(e1)
    e1 = Activation('elu', name='e_3')(e1)

    e2 = Conv1D(filters=256, strides=1, kernel_size=1, name='e_7')(e1)
    e2 = BatchNormalization(name='e_8')(e2)
    e2 = Activation('elu', name='e_9')(e2)

    e2 = Add()([e1, e2])

    e3 = Conv1D(filters=2 * n, strides=1, kernel_size=1, name='e_10')(e2)  # 2 = I and Q channels
    e3 = BatchNormalization(name='e_11')(e3)
    e3 = Activation('linear', name='e_12')(e3)

    c = Lambda(normalization, name='power_norm')(e3)
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c,backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=256, strides=1, kernel_size=1, name='d_1')(c)
    d1 = BatchNormalization(name='d_2')(d1)
    d1 = Activation('elu', name='d_3')(d1)

    d2 = Conv1D(filters=256, strides=1, kernel_size=1, name='d_7')(d1)
    d2 = BatchNormalization(name='d_8')(d2)
    d2 = Activation('elu', name='d_9')(d2)

    d2 = Add()([d1, d2])

    # Output One hot vector and use Softmax to soft decoding
    model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d2)

    # Build System Model
    model = Model(model_input, model_output)
    return model

def model3(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    c = Lambda(normalization, name='power_norm')(e1)
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c,backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

    # Define Decoder Layers (Receiver)
    d1 = Conv1D(filters=256, strides=1, kernel_size=1, name='d_1')(c)
    d1 = BatchNormalization(name='d_2')(d1)
    d1 = Activation('elu', name='d_3')(d1)

    d2 = Conv1D(filters=256, strides=1, kernel_size=1, name='d_7')(d1)
    d2 = BatchNormalization(name='d_8')(d2)
    d2 = Activation('elu', name='d_9')(d2)

    d2 = Concatenate()([d1, d2])

    # Output One hot vector and use Softmax to soft decoding
    model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d2)

    # Build System Model
    model = Model(input, model_output)
    return model

def model4(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    model_input = Input(batch_shape=(None, L, 2 ** k), name='input_bits')

    e1 = Dense(2 * n)(model_input)

    c = Lambda(normalization, name='power_norm')(e1)
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c,backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

    # Define Decoder Layers (Receiver)
    d0 = Conv1D(filters=256, strides=1, kernel_size=1,)(c)
    d0 = BatchNormalization()(d0)
    d0 = Activation('elu')(d0)

    d1 = Conv1D(filters=256, strides=1, kernel_size=1,)(d0)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d1 = Concatenate()([d0, d1])

    d2 = Conv1D(filters=256, strides=1, kernel_size=1, name='e_1')(d1)
    d2 = BatchNormalization(name='e_2')(d2)
    d2 = Activation('elu', name='e_3')(d2)

    d2 = Concatenate()([d0, d1, d2])

    model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d2)

    # Build System Model
    model = Model(model_input, model_output)
    return model


def model2_first(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    num_fiters_1 = 64
    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    c = Lambda(normalization, name='power_norm')(e1)

    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c, backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

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

def model3_first(channel, L, k, n, batch_size, noise_sigma, backendNoise, burst_beta):
    num_fiters_1 = 64
    num_fiters_2 = 16

    kernel_size_1 = 32
    kernel_size_2 = 4
    kernel_size_3 = 1

    input = Input(batch_shape=(batch_size, L, 2 ** k))  # batch_size, length, channels
    e1 = Dense(2 * n)(input)

    # Tx Power normalization
    c = Lambda(normalization, name='power_norm')(e1)
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c, backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

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


def model5(channel, L, k, n, noise_sigma, backendNoise, burst_beta):
    model_input = Input(batch_shape=(None, L, 2 ** k), name='input_bits')

    e1 = Dense(2 * n)(model_input)

    c = Lambda(normalization, name='power_norm')(e1)
    if channel == 'awgn':
        c = awgn(c, noise_sigma)
    elif channel == 'bursty':
        c = bursty(c,backendNoise, burst_beta)
    else:
        c = rayleigh(c, noise_sigma)

    # Define Decoder Layers (Receiver)
    d0 = Conv1D(filters=256, strides=1, kernel_size=1,)(c)
    d0 = BatchNormalization()(d0)
    d0 = Activation('elu')(d0)

    d1 = Conv1D(filters=256, strides=1, kernel_size=1,)(d0)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d1 = Add()([d0, d1])

    d2 = Conv1D(filters=256, strides=1, kernel_size=1, name='e_1')(d1)
    d2 = BatchNormalization(name='e_2')(d2)
    d2 = Activation('elu', name='e_3')(d2)

    d2 = Add()([d0, d1, d2])

    model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d2)

    # Build System Model
    model = Model(model_input, model_output)
    return model
