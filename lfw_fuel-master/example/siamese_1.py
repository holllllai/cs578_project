from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Lambda 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
from keras import backend as K
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

batch_size = 128
nb_epoch = 12
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

print(y_train[:30])
print(y_train[-30:])
print(y_test[:30])
print(y_test[-30:])

train_pairs = np.zeros((2, 4400, feature_width, feature_height, 3))



train_pairs[0] = X_train[:,:,:,0:3]
train_pairs[1] = X_train[:,:,:,3:6]

# print(train_pairs.shape)
# print(train_pairs[0].shape)

test_pairs = np.zeros((2, 2000, feature_width, feature_height, 3))

test_pairs[0] = X_test[:,:,:,0:3]
test_pairs[1] = X_test[:,:,:,3:6]

input_shape = (feature_width, feature_height, 3)

# def euclidean_distance(inputs):
#     assert len(inputs) == 2, \
#         'Euclidean distance needs 2 inputs, %d given' % len(inputs)
#     u, v = inputs
#     return K.sqrt((K.square(u - v)).sum(axis=1, keepdims=True))
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels, threshold):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < threshold].mean()

def compute_real_accuracy(predictions, labels, threshold):
    match_count = 0
    for i in range(len(predictions)):
        if predictions[i] < threshold and labels[i]==1:
            match_count += 1
        if predictions[i] > threshold and labels[i]==0:
            match_count += 1
    return float(match_count)/len(predictions)



def create_base_network(input_shape):
    net = Sequential()
    net.add(Conv2D(32, (5,5), input_shape=input_shape, padding='same', data_format='channels_last', activation='relu'))
    # net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
    # net.add(Dropout(0.2))

    net.add(Conv2D(64, (6,6), padding='same', data_format='channels_last', activation='relu'))
    # net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(5,5), data_format='channels_last'))
    #net.add(Dropout(0.3))
    net.add(Conv2D(128, (5,5), padding = 'same', data_format = 'channels_last', activation = 'relu'))


    # net.add(Conv2D(128, (3,3), padding='same', data_format='channels_last', activation='relu'))
    # net.add(BatchNormalization())
    # net.add(MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
    # net.add(Dropout(0.3))
    net.add(Flatten())
    net.add(Dense(64, activation='relu'))# kernel_regularizer = regularizers.l2(0.01)))
    net.add(BatchNormalization())
    net.add(Dropout(0.4))
    # net.add(Dense(128,activation='relu'))
    return net



input_a = Input(input_shape)
input_b = Input(shape=input_shape)
base_network = create_base_network(input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)



adam = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(loss = contrastive_loss, optimizer=adam)
history = model.fit([train_pairs[0], train_pairs[1]], y_train, batch_size = batch_size, epochs = nb_epoch, verbose=1, validation_data=([test_pairs[0], test_pairs[1]], y_test))
pred = model.predict([train_pairs[0], train_pairs[1]])
print(pred[:30])
print(pred[-30:])

thresholds = [0.3,0.4,0.5,0.6,0.7]
for threshold in thresholds:
    acc= compute_accuracy(pred, y_train, threshold)
    new_acc = compute_real_accuracy(pred, y_train, threshold)
    print('-'*40)
    print('Accuracy with threshold ' + str(threshold) +' on training set: %0.2f%%' % (100 * acc))
    print('Real Accuracy with threshold ' + str(threshold) +' on training set: %0.2f%%' % (100 * new_acc))

threshold = 0.5

# tr_acc = compute_accuracy(pred, y_train,threshold)
pred = model.predict([test_pairs[0], test_pairs[1]])
print(pred[:30])
print(pred[-30:])
te_acc = compute_accuracy(pred, y_test, threshold)
te_real_acc = compute_real_accuracy(pred, y_test, threshold)


print(history.history.keys())
# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print ('-'*40)
print('Accuracy with threshold ' + str(threshold) +' on test set: %0.2f%%' % (100 * te_acc))
print('Real Accuracy with threshold ' + str(threshold) +' on test set: %0.2f%%' % (100 * te_real_acc))
# for x in range(len(pred)):
    # print('pred is ' + str(pred[i]) + ' and label is ' + str(y_test[i]))

# print (np.sum(y_test))
# print (pred.shape)
# print (np.where( pred < 2.5))

# count = 0
# for i in range(len(pred)):
#     if pred[i]<threshold:
#         count+=1
# print ('Of all test data points, ' + str(count) + ' of them are labeled as matched pairs')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# print(history.history.keys())
# plt.plot(history.history['binary_accuracy'])
# plt.plot(history.history['val_binary_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
