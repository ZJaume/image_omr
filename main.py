from PIL import Image, ImageOps
import numpy as np
import glob
import models

from keras.models import Model
import keras.backend as K

nb_classes = 216
nb_epoch = 30
batch_size = 128

pool_size = 2

# Max length of an example label
lb_max_length = 15

# Size of the images
img_rows, img_cols = 180, 80

#
# Load data function, recieves downsample factor equal to pool size
#
def load_data(downsample_factor):
    image_list = []
    class_list = []
    label_length = []

    for directory in {'TestSet', 'TrainSet'}:
        path = './data/lilypond/{}/'.format(directory)
        labels = open(path + 'labels_cod.txt')
        for filename in glob.glob(path + '*.png'):
            im=Image.open(filename).resize((img_rows,img_cols)).convert('L')
            im=ImageOps.invert(im)      # Meaning of grey level is 255 (black) and 0 (white)
            label = np.fromstring(labels.readline(), dtype=int, sep=' ')
            label_length.append(len(label))
            class_list.append(label)
            image_list.append(np.asarray(im).astype('float32')/255)

    n = len(image_list)     # Total examples
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
        X = np.asarray(image_list).reshape(n, 1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
        X = np.asarray(image_list).reshape(n, img_rows, img_cols, 1)

    inputs = { 'the_input' : X,
                'the_labels' : np.asarray(class_list),
                'input_length' : np.full((n,), img_cols // downsample_factor ** 2 - 2),
                'label_length' : np.asarray(label_length),
            }
    outputs = {'ctc' : np.zeros((n,), dtype=int)}

    return inputs, outputs, input_shape

X, Y, input_shape = load_data(pool_size)

print(str(Y['ctc'].shape[0]) + " trainning examples")
print(str(nb_epoch) + " epochs")

model = models.create_rnn(input_shape, lb_max_length, nb_classes)
model.fit(X, Y['ctc'], batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)
