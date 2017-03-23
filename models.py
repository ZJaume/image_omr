from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, merge
from keras.models import Sequential, Model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD, RMSprop, Nadam
from keras import backend as K
import keras.callbacks

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
    rnn_size = 128
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
    gru_1 = Bidirectional(GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1'))(inner)
    gru_2 = Bidirectional(GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2'))(gru_1)

    # transforms RNN output to character activations:
    inner = Dense(nb_classes, init='he_normal',
                  name='dense2')(gru_2)
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

class AccCallback(keras.callbacks.Callback):

    def __init__(self,test_func, inputs):
        self.test_func = test_func
        self.inputs = inputs

    def on_epoch_end(self, epoch, logs={}):
        func_out = self.test_func([self.inputs['the_input']])[0]
        ed = 0
        mean_ed = 0.0
        mean_norm_ed = 0.0
        for i in range(func_out.shape[0]):
            output = []
            prev = -1
            for j in range(func_out.shape[1]):
                out = np.argmax(func_out[i][j])
                #print(func_out[i][j])
                if out != prev and out != -1:
                    output.append(out)
                prev = out
            print(output)
            print(str(self.inputs['the_labels'][i])+str(self.inputs['label_length'][i]))
            ed = self.levenshtein(self.inputs['the_labels'][i].tolist(),output)
            mean_ed += float(ed)
            mean_norm_ed += float(ed) / self.inputs['label_length'][i]

        mean_ed = mean_ed / len(func_out)
        mean_norm_ed = mean_norm_ed / len(func_out)
        print("Mean edit distance: %0.3f, mean normalized edit distance: %0.3f" % (mean_ed, mean_norm_ed))

    def levenshtein(self,a,b):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n

        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]
