from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, merge, Permute
from keras.models import Sequential, Model
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Nadam
from keras import backend as K

import numpy as np

# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#
# Create a convolution block and add to a tensor
# recives a tuple with the configuration (nb_convs, nb_filters, filter_size, pool_size, batch_normalization)
#
def conv_block(input, config, name='1'):
    for i in range(config[0]):
        input = Convolution2D(config[1], config[2], config[2], border_mode='same',
            activation='relu', name='conv'+ name + '_' + str(i+1))(input)
    input = MaxPooling2D(pool_size=config[3], name='max'+name)(input)
    return input

#
# Creates a block of bidirectional RNN with GRU
# config is (num_layers, num_units)
#
def rnn_block(input, config):
    for i in range(config[0]):
        input = Bidirectional(GRU(config[1], return_sequences=True, name='gru'+str(i)))(input)
    return input

#
# Create RNN with convolutional filters and CTC logloss function
#
def create_rnn(input_shape, lb_max_length, nb_classes, config, pool_size=2):
    if K.image_dim_ordering() =='th':
        img_h = input_shape[1]
        img_w = input_shape[2]
    else:
        img_h = input_shape[0]
        img_w = input_shape[1]

    # Network parameters
    nb_filters1 = 64
    nb_filters2 = 128
    nb_filters3 = 256
    nb_filters4 = 512

    filter_size1 = 3
    filter_size2 = 3
    filter_size3 = 3
    filter_size4 = 3

    pool1 = (pool_size, pool_size)
    pool2 = (pool_size, 1)

    rnn_size = config[1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    # Convolution block 1
    inner = conv_block(input_data, (1, nb_filters1, filter_size1, pool1), name='1')

    # Convolution block 4
    inner = conv_block(inner, (1, nb_filters2, filter_size2, pool1), name='2')

    # Convolution block 3
    inner = conv_block(inner, (2, nb_filters3, filter_size3, pool2), name='3')

    # Convolution block 4
    inner = conv_block(inner, (2, nb_filters4, filter_size4, pool2), name='4')

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 4)) * nb_filters4)
    inner = Permute((3,1,2))(inner)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    #inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru = rnn_block(inner, (config[0],rnn_size))

    # transforms RNN output to character activations:
    inner = Dense(nb_classes, init='he_normal',
                  name='dense2')(gru)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(input=[input_data], output=y_pred).summary()

    labels = Input(name='the_labels', shape=[lb_max_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='rmsprop')

    test_func = K.function([input_data],[y_pred])

    return model, test_func

