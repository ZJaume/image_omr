from PIL import Image, ImageOps
import numpy as np
import glob
import models
import sys
import os
import json

from keras.models import Model, save_model, load_model
from acc_callback import AccCallback
import keras.backend as K

nb_epoch = 20
batch_size = 128
super_batch = 10000

pool_size = 2

# Max length of an example label
lb_max_length = 8

# Size of the images
img_w, img_h= 120, 32

# Read the filepath and the classes
if len(sys.argv) <= 1:
    print("ERROR: the first argument must be the dataset path")
    sys.exit(0)
if not os.path.isdir(sys.argv[1]):
    print("ERROR: the first argument must be an existing directory")
    sys.exit(0)
path = sys.argv[1]
nb_classes = len(json.load(open(path + "dictionary.json", 'r')))
print("Number of classes: %d " % nb_classes)

#
# Load data function, recieves downsample factor equal to pool size
#
def load_data(downsample_factor, paths, labels):
    image_list = []
    class_list = []
    label_length = []
    num_examples = batch_size*90

    for filename, lb in zip(paths, labels):
        im=Image.open(filename).resize((img_w,img_h)).convert('L')
        im=ImageOps.invert(im)      # Meaning of grey level is 255 (black) and 0 (white)
        label = np.fromstring(lb, dtype=int, sep=' ')
        label_length.append(len(label))
        fill = np.full((lb_max_length - len(label),), -1, dtype=int)
        class_list.append(np.append(label,fill))
        image_list.append(np.asarray(im).astype('float32'))#/255)
        #num_examples-=1

    n = len(image_list)     # Total examples
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_h, img_w)
        X = np.asarray(image_list).reshape(n, 1, img_h, img_w)
    else:
        input_shape = (img_h, img_w, 1)
        X = np.asarray(image_list).reshape(n, img_h, img_w, 1)
    class_list = np.asarray(class_list)
    label_length = np.asarray(label_length)

    # Normalize
    mean_image = np.mean(X,axis=0)
    X -= mean_image
    X /= 128

    inputs= { 'the_input' : X,
                'the_labels' : class_list,
                'input_length' : np.full((n,), img_w // downsample_factor ** 2 - 2),
                'label_length' : label_length,
                }
    outputs= {'ctc' : np.zeros((n,), dtype=int)}

    return inputs, outputs, input_shape

def sort_paths(num_paths, prefix, suffix, offset=0):
    sorted_paths = []
    for i in range(num_paths):
        sorted_paths.append(prefix + str(i+1) + suffix)
    return sorted_paths

def shuffle(a, b):
    randomize = np.arange(a.shape[0])
    np.random.shuffle(randomize)
    return a[randomize], b[randomize]

#
# Perform trainning dividing each epoch in batches
#
def train_super_epoch(paths, labels, n_partition):
    # Load test examples
    X_test, Y_test, input_shape = load_data(pool_size, paths[n_partition:], labels[n_partition:])

    model, test_func = models.create_rnn(input_shape, lb_max_length, nb_classes+1)
    acc_callback = AccCallback(test_func, X_test, nb_classes, batch_size, logs=True, name='plot_acc_drop_05_05')

    for i in range(nb_epoch):
        print("Super epoch {}/{}".format(i+1,nb_epoch))
        j = 0
        while j < n_partition:
            if j+super_batch < n_partition:
                X_train, Y_train, input_shape = load_data(pool_size, paths[j:j+super_batch], labels[j:j+super_batch])
            else:
                X_train, Y_train, input_shape = load_data(pool_size, paths[j:n_partition], labels[j:n_partition])
            print("\t-Trainning from {} to {}".format(j,j+X_train['the_input'].shape[0]))
            hist = model.fit(X_train, Y_train['ctc'], batch_size=batch_size, nb_epoch=1, verbose=0)
            print("\t  Loss: %0.3f" % hist.history['loss'][0])
            print("")
            j += super_batch
        acc_callback.on_epoch_end(i)
    acc_callback.on_train_end()
    return model

fp = open(path + 'labels_cod.txt', 'r')
labels = fp.read().split('\n')
if labels[-1] == '':
    labels.pop()
fp.close()
num_paths = len(glob.glob(path + '*.png'))
paths = np.asarray(sort_paths(num_paths, path, '.png'))
labels = np.asarray(labels)
# Calculate de max length label
for lb in labels:
    label = np.fromstring(lb, dtype=int, sep=' ')
    lb_max_length = max(lb_max_length,len(label))
print("Max length label: %d" % lb_max_length)

# Divide in train and test data
n_partition = int(num_paths*0.9)    # 10% validation
paths, labels = shuffle(paths, labels)
print("%d training examples" % n_partition)
print("%d test examples" % (num_paths-n_partition))
print("%d epochs" % nb_epoch)

train_super_epoch(paths, labels, n_partition)
'''
X_train, Y_train, input_shape = load_data(pool_size, paths[:n_partition], labels[:n_partition])
X_test, Y_test, input_shape = load_data(pool_size, paths[n_partition:], labels[n_partition:])

print(str(len(X_train['the_input'])) + " trainning examples")
print(str(len(X_test['the_input'])) + " test examples")
print(str(nb_epoch) + " epochs")

# nb_classes+1 for the ctc blank class
model, test_func = models.create_rnn(input_shape, lb_max_length, nb_classes+1)
acc_callback = AccCallback(test_func, X_test, nb_classes, batch_size, logs=True)

model.fit(X_train, Y_train['ctc'], batch_size=batch_size, nb_epoch=nb_epoch,
        callbacks=[acc_callback], validation_data=(X_test,Y_test['ctc']))
        '''
#model.load_weights("lilypond_rnn-w.h5")
#out = test_func([X['the_input'][1:2]])[0]
#print(X['the_labels'][1:2])
#print(out.shape)
#for f in range(out[0].shape[0]):
#    print("Frame :"+str(f)+" "+ str(np.argmax(out[0][f])))
