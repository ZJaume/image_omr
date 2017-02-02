from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, merge
from keras.models import Sequential, Model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD, Nadam
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
def create_rnn(input_shape, lb_max_length, nb_classes, pool_size=2):
    if K.image_dim_ordering() =='th':
        img_w = input_shape[1]
        img_h = input_shape[2]
    else:
        img_w = input_shape[0]
        img_h = input_shape[1]

    # Network parameters
    conv_num_filters = 16
    filter_size = 3
    time_dense_size = 32
    rnn_size = 512
    output_size = 28
    act = 'relu'

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_num_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(inner)
    gru1_merged = merge([gru_1, gru_1b], mode='sum')
    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(nb_classes, init='he_normal',
                  name='dense2')(merge([gru_2, gru_2b], mode='concat'))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(input=[input_data], output=y_pred).summary()

    labels = Input(name='the_labels', shape=[lb_max_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    return model

