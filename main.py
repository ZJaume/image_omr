from PIL import Image, ImageOps
import numpy as np
import glob
import models
import os

from keras.models import Model, save_model, load_model
import keras.backend as K

nb_classes = 215
nb_epoch = 50
batch_size = 128

pool_size = 2

# Max length of an example label
lb_max_length = 15

# Size of the images
img_w, img_h= 120, 32

#
# Load data function, recieves downsample factor equal to pool size
#
def load_data(downsample_factor):
    image_list = []
    class_list = []
    label_length = []
    num_examples = batch_size*90

    for directory in ['TrainSet', 'TestSet']:
        path = './data/lilypond/{}/'.format(directory)
        num_paths = len(glob.glob(path + '*.png'))
        paths = sort_paths(num_paths, len(image_list), path, '.png')
        labels = open(path + 'labels_cod.txt')

        for filename in paths:
            im=Image.open(filename).resize((img_w,img_h)).convert('L')
            im=ImageOps.invert(im)      # Meaning of grey level is 255 (black) and 0 (white)
            label = np.fromstring(labels.readline(), dtype=int, sep=' ')
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

    # Using less examples
    # X = X[num_examples:]
    # class_list = class_list[num_examples:]
    # label_length = label_length[num_examples:]
    # n = num_examples

    # Divide in train and test data
    randomize = np.arange(n)
    np.random.shuffle(randomize)
    X, class_list, label_length = X[randomize], class_list[randomize], label_length[randomize]
    n_partition = int(n*0.9)    # 10% validation

    inputs_train = { 'the_input' : X[:n_partition],
                'the_labels' : class_list[:n_partition],
                'input_length' : np.full((n_partition,), img_w // downsample_factor ** 2 - 2),
                'label_length' : label_length[:n_partition],
                }
    inputs_test = { 'the_input' : X[n_partition:],
                'the_labels' : class_list[n_partition:],
                'input_length' : np.full((n-n_partition,), img_w // downsample_factor ** 2 - 2),
                'label_length' : label_length[n_partition:],
                }
    outputs_train = {'ctc' : np.zeros((n_partition,), dtype=int)}
    outputs_test = {'ctc' : np.zeros((n-n_partition,), dtype=int)}

    return inputs_train, inputs_test, outputs_train, outputs_test, input_shape

def sort_paths(num_paths, offset, prefix, suffix):
    sorted_paths = []
    for i in range(1+offset, num_paths+offset+1):
        sorted_paths.append(prefix + str(i) + suffix)
    return sorted_paths

X_train, X_test, Y_train, Y_test, input_shape = load_data(pool_size)

print(str(len(X_train['the_input'])) + " trainning examples")
print(str(len(X_test['the_input'])) + " test examples")
print(str(nb_epoch) + " epochs")
print(X_test['the_input'][0])

# nb_classes+1 for the ctc blank class
model, test_func = models.create_rnn(input_shape, lb_max_length, nb_classes+1)
acc_callback = models.AccCallback(test_func, X_test, nb_classes, logs=True)

model.fit(X_train, Y_train['ctc'], batch_size=batch_size, nb_epoch=nb_epoch,
        callbacks=[acc_callback], validation_data=(X_test,Y_test['ctc']))
#model.load_weights("lilypond_rnn-w.h5")
#out = test_func([X['the_input'][1:2]])[0]
#print(X['the_labels'][1:2])
#print(out.shape)
#for f in range(out[0].shape[0]):
#    print("Frame :"+str(f)+" "+ str(np.argmax(out[0][f])))
