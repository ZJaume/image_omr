from PIL import Image, ImageOps
import numpy as np
import glob
import models

from keras.models import Model
import keras.backend as K

nb_classes = 216
nb_epoch = 30
batch_size = 128

# Max length of an example label
lb_max_length = 16

# Size of the images
img_rows, img_cols = 180, 80

def load_data():
    image_list = []
    class_list = []

    for directory in {'TestSet', 'TrainSet'}:
        path = './data/lilypond/{}/'.format(directory)
        labels = open(path + 'labels_cod.txt')
        for filename in glob.glob(path + '*.png'):
            im=Image.open(filename).resize((img_rows,img_cols)).convert('L')
            im=ImageOps.invert(im)      # Meaning of grey level is 255 (black) and 0 (white)
            image_list.append(np.asarray(im).astype('float32')/255)
            class_list.append(np.fromstring(labels.readline(), dtype=int, sep=' '))

    n = len(image_list)     # Total examples
    if K.image_dim_ordering() == 'th':
        X = np.asarray(image_list).reshape(n,1,img_rows,img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = np.asarray(image_list).reshape(n,img_rows,img_cols,1)
        input_shape = (img_rows, img_cols, 1)
    Y = np.asarray(class_list)

    return X, Y, input_shape

X, Y, input_shape = load_data()

print(str(len(Y)) + " trainning examples")
print(str(nb_epoch) + " epochs")

model = models.create_rnn(input_shape, lb_max_length, nb_classes)
model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)
