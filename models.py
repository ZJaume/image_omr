from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD
from keras import backend as K

# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#
# Create RNN with convolutional filters and CTC logloss function
#
def create_rnn(input_shape):
    if K.image_dim_ordering() =='th':
        img_w = input_shape[1]
        img_h = input_shape[2]
    else:
        img_w = input_shape[2]
        img_h = input_shape[1]

    # Network parameters
    conv_num_filters = 16
    filter_size = 3
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    output_size = 28
    act = 'relu'

    model = Sequential()
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                activation=act, init='he_normal', name='conv1', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='max1'))
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                activation=act, init='he_normal', name='conv2'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='max2'))

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_num_filters)
    model.add(Reshape(target_shape=conv_to_rnn_dims, name='reshape'))
    model.add(Dense(time_dense_size, activation=act, name='dense1'))

    model.add(Bidirectional(GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')))
    model.add(Bidirectional(GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')))

    model.add(Dense(output_size, init='he_normal',name='dense2'))
    model.add(Activation('softmax', name='softmax'))

    model.add(Lambda(ctc_lambda_func, output_shape=(1,), name='ctc'))
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss='ctc', optimizer=sgdi, metricss=['accuracy'])

    model.summary()
    return model

model = create_rnn((1,457,136))
