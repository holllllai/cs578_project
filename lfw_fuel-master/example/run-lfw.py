from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from scipy.misc import imresize 
import matplotlib.pylab as plt
# this allows the example to be run in-repo
# (or can be removed if lfw_fuel is installed)
import sys
sys.path.append('.')

from lfw_fuel import lfw

'''
    Train a simple convnet on the LFW dataset.
'''

batch_size = 64
nb_epoch = 50
feature_width = 64
feature_height = 64
downsample_size = 64

def crop_and_downsample(originalX, downsample_size):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim)/2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = originalX[:, left_margin:right_margin, left_margin:right_margin]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    resized1 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
    resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # re-packge into a new X entry
    newX = np.concatenate([resized1,resized2], axis=2)

    # the next line is important.
    # if you don't normalize your data, all predictions will be 0 forever.
    newX = newX/255.0

    return newX

(X_train, y_train), (X_test, y_test) = lfw.load_data("deepfunneled")

# print(y_train[:20])

# the data, shuffled and split between train and test sets
X_train = np.asarray([crop_and_downsample(x,downsample_size) for x in X_train])
X_test  = np.asarray([crop_and_downsample(x,downsample_size) for x in X_test])
# print(X_train.shape)
# print(X_test.shape)


X_train_img1_flipped = np.flip(X_train[:,:,:,0:3],2)

X_train_img2_flipped = np.flip(X_train[:,:,:,3:6],2)
X_train_extra = np.concatenate( (X_train_img1_flipped, X_train_img2_flipped), axis = 3)


# print (X_train_extra.shape)

X_train = np.concatenate( (X_train, X_train_extra), axis=0)
y_train = np.concatenate((y_train,y_train),axis = 0)

X_test_img1_flipped = np.flip(X_test[:,:,:,0:3],2)
X_test_img2_flipped = np.flip(X_test[:,:,:,3:6],2)
X_test_extra = np.concatenate( (X_test_img1_flipped, X_test_img2_flipped), axis = 3)
X_test = np.concatenate((X_test, X_test_extra), axis =0 )
y_test = np.concatenate((y_test, y_test), axis = 0)
# print(X_train.shape)
# print(y_train.shape)
# fig = plt.figure()
# ax1,ax2= [fig.add_subplot(1,2,i+1) for i in range(2)]

# ax1.imshow(X_train[1,:,:,0:3])
# ax2.imshow(X_train[1,:,:,3:6])
# plt.show()

# ax3,ax4 = [fig.add_subplot(1,2,i+1) for i in range(2)]

# ax3.imshow(X_train[2201,:,:,0:3])
# ax4.imshow(X_train[2201,:,:,3:6])

# plt.show()


# ax1.imshow(np.fliplr(X_train[5,:,:,0:3]))
# plt.show()

model = Sequential()



model.add(Conv2D(32, (5,5), input_shape=(downsample_size,downsample_size,6), padding='same', data_format='channels_last', activation='relu'))
model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))


model.add(Conv2D(32, (5,5), padding='same', data_format='channels_last', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))


model.add(Conv2D(64, (5,5), padding='same', data_format='channels_last', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))# kernel_regularizer = regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))#,kernel_regularizer = regularizers.l2(0.01)))
adam = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=adam)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


# model = Sequential()

# model.add(Conv2D(32, (5,5), input_shape=(downsample_size,downsample_size,6), padding='same', data_format='channels_last', activation='relu'))
# model.add(Conv2D(32, (5,5), padding='same', data_format='channels_last', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
# model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# model = Sequential()
# #1st layer
# model.add(Conv2D(filters = 32, input_shape = (downsample_size,downsample_size,6), kernel_size = (5,5), strides = (1,1),padding = 'valid'))
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))

# #2nd layer
# model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = (1,1), padding = 'valid'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2), padding = 'valid'))

#3rd layer
# model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = (1,1), padding = 'valid'))
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
#4th layer
# model.add(Amodel.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
# ctivation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
# model.add(Dropout(0.5))
# #FC
# model.add(Flatten())
# #1st FC 
# model.add(Dense(1024, input_shape=(32*32*6,)))
# model.add(Activation('relu'))
# #dropout
# model.add(Dropout(0.4))

#2nd FC
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))

#3rd FC
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))

#output layer
# model.add(Dense(1))
# model.add(Activation('softmax'))

# model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
# model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

