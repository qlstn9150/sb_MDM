import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from funcs_v1 import *

#import tensorflow as tf  # tensorflow 1.x
import tensorflow.compat.v1 as tf  # tensorflow 2.x
tf.disable_v2_behavior()  # tensorflow 2.x
tf.reset_default_graph()


'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
train_data = np.random.randint(low=0, high=2, size=(nb_train_word, k * L))
# Used as labeled data
label_data = copy.copy(train_data)
train_data = np.reshape(train_data, newshape=(nb_train_word, L, k))

# Convert Binary Data to integer
tmp_array = np.zeros(shape=k)
for i in range(k):
    tmp_array[i] = 2 ** i
int_data = tmp_array[::-1]

# Convert Integer Data to one-hot vector
int_data = np.reshape(int_data, newshape=(k, 1))
one_hot_data = np.dot(train_data, int_data)
vec_one_hot = to_categorical(y=one_hot_data, num_classes=2 ** k)

# used as Label data
label_one_hot = copy.copy(vec_one_hot)

'''
 --- NEURAL NETWORKS PARAMETERS ---
'''

early_stopping_patience = 10

epochs = 5  # 150

optimizer = Adam(lr=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)

# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)

# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'model_LBC_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '_v1.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)


###########################################
#model = model_dnn(noise_sigma)
model = model_cnn(noise_sigma)
#model = model_cnn_gru(noise_sigma)
#model = model_densenet(noise_sigma)
#model = model_resnet(noise_sigma)
#model = model_gru(noise_sigma)
#model = model13()
###########################################


# %%
# Print Model Architecture
model.summary()

# Compile Model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# print('encoder output:', '\n', encoder.predict(vec_one_hot, batch_size=batch_size))

print('starting train the NN...')
start = time.process_time()

# TRAINING
mod_history = model.fit(vec_one_hot, label_one_hot,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.3, callbacks=[modelcheckpoint, reduce_lr])

end = time.process_time()

print('The NN has trained ' + str(end - start) + ' s')

# Plot the Training Loss and Validation Loss
hist_dict = mod_history.history

val_loss = hist_dict['val_loss']
loss = hist_dict['loss']
acc = hist_dict['acc']
# val_acc = hist_dict['val_acc']
print('loss:', loss)
print('val_loss:', val_loss)

epoch = np.arange(1, epochs + 1)

plt.semilogy(epoch, val_loss, label='val_loss')
plt.semilogy(epoch, loss, label='loss')

plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')

plt.show()