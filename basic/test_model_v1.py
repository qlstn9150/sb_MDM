from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, SimpleRNN, LSTM, \
    GRU, Bidirectional, TimeDistributed, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as KR
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

#import tensorflow as tf  # tensorflow 1.x
import tensorflow.compat.v1 as tf  # tensorflow 2.x
tf.disable_v2_behavior()  # tensorflow 2.x
tf.reset_default_graph()

from funcs_v1 import *



# Initial Vectors
Vec_Eb_N0 = []
Bit_error_rate = []

'''
 --- GENERATING INPUT DATA ---
'''

# Initialize information Data 0/1
in_sym = np.random.randint(low=0, high=2, size=(num_of_sym, k * L))
label_data = copy.copy(in_sym)
in_sym = np.reshape(in_sym, newshape=(num_of_sym, L, k))

# Convert Binary Data to integer
tmp_array = np.zeros(shape=k)
for i in range(k):
    tmp_array[i] = 2 ** i
int_data = tmp_array[::-1]

# Convert Integer Data to one-hot vector
int_data = np.reshape(int_data, newshape=(k, 1))
one_hot_data = np.dot(in_sym, int_data)
vec_one_hot = to_categorical(y=one_hot_data, num_classes=2 ** k)

# used as Label data
label_one_hot = copy.copy(vec_one_hot)


# Define Channel Layers including AWGN and Flat Rayleigh fading
def channel_layer(x, sigma):
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)

    return x + w


def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = number of NN into the channel


print('start simulation ...' + str(k) + '_' + str(L) + '_' + str(n))

'''
 --- DEFINE THE Neural Network(NN) ---
'''

# Eb_N0 in dB
for Eb_N0_dB in range(2, 9, 1):
    # Noise Sigma at this Eb

    noise_sigma = np.sqrt(1 / (2 * R * 10 ** (Eb_N0_dB / 10)))


    ###########################################
    #model = model_dnn(noise_sigma)
    model = model_cnn(noise_sigma)
    # model = model_cnn_gru(noise_sigma)
    # model = model_densenet(noise_sigma)
    # model = model_resnet(noise_sigma)
    # model = model_gru(noise_sigma)
    #model = model13()
    ###########################################


    # Load Weights from the trained NN
    model.load_weights('./' + 'model_LBC_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '_v1.h5',
                       by_name=False)

    '''
    RUN THE NN
    '''

    # %%
    # RUN Through the Model and get output
    decoder_output = model.predict(vec_one_hot, batch_size=batch_size)

    '''
     --- CALULATE BLER ---

    '''

    # Decode One-Hot vector
    position = np.argmax(decoder_output, axis=2)
    tmp = np.reshape(position, newshape=one_hot_data.shape)
    error_rate = np.mean(np.not_equal(one_hot_data, tmp))

    print('Eb/N0 = ', Eb_N0_dB)
    print('BLock Error Rate = ', error_rate)

    print('\n')

    # Store The Results
    Vec_Eb_N0.append(Eb_N0_dB)
    Bit_error_rate.append(error_rate)

'''
PLOTTING
'''
# Print BER
# print(Bit_error_rate)

print(Vec_Eb_N0, '\n', Bit_error_rate)

with open('BLER_model_LBC_' + str(k) + '_' + str(n) + '_' + str(L) + '_AWGN' + '_v1.txt', 'w') as f:
    print(Vec_Eb_N0, '\n', Bit_error_rate, file=f)
f.closed

# Plot BER Figure
plt.semilogy(Vec_Eb_N0, Bit_error_rate, color='red')
label = [str(k) + '_' + str(L)]
plt.legend(label, loc=0)
plt.xlabel('Eb/N0')
plt.ylabel('BER')
plt.title(str(k) + '_' + str(n) + '_' + str(L))
plt.grid('true')
plt.show()