import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow.compat.v1 as tf  # tensorflow 2.x
tf.disable_v2_behavior()  # tensorflow 2.x
tf.reset_default_graph()

from model import *

def test(model_str, model_f, channel, L=8, k=4, n=4, batch_size=64):

    Vec_Eb_N0 = []
    Bit_error_rate = []
    one_hot_data, vec_one_hot, label_one_hot = generate_data(1000, k, L, batch_size)

    if channel == 'awgn':
        range = list(range(2,9))
    elif channel == 'bursty':
        range = range(0,21,5)
    else:
        range = range(0,31,5)

    for Eb_N0_dB in range:

        R = k / n
        noise_sigma = np.sqrt(1 / (2 * R * 10 ** (Eb_N0_dB / 10)))
        backendNoise = 1 / (2 * R * 10 ** (Eb_N0_dB / 10))
        burst_beta = np.random.binomial(1, alpha=0.05, size=(batch_size, L, 2 * n))

        model = model_f(channel, noise_sigma, backendNoise, burst_beta)
        model.load_weights('./' + channel + '/' + model_str + '.h5', by_name=False)

        decoder_output = model.predict(vec_one_hot, batch_size=batch_size)

        # BER
        position = np.argmax(decoder_output, axis=2)
        tmp = np.reshape(position,newshape=one_hot_data.shape)
        error_rate = np.mean(np.not_equal(one_hot_data,tmp))

        Vec_Eb_N0.append(Eb_N0_dB)
        Bit_error_rate.append(error_rate)
        print('Eb/N0 = ', Eb_N0_dB)
        print('BLock Error Rate = ', error_rate)
        print('\n')

    with open('./' + channel + '/' + model_str + '.txt', 'w') as f:
        print(Vec_Eb_N0, '\n', Bit_error_rate, file=f)
    f.closed


test(model_str = 'basic',  model_f = basic,
      channel='awgn')