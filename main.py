from PIL import Image, ImageOps
import numpy as np
import glob
import models

nb_classes = 215

# Size of the images
img_rows, img_cols = 180, 80

def load_data():
    image_list = []
    class_list = []

    for directory in {'TestSet', 'TrainSet'}:
        path = './data/lilypond/{}/'.format(directory)
        labels = open(path + 'labels_cod.txt')
        class_list.append(np.fromstring(labels.readline(), dtype=int, sep=' ')
        for filename in glob.glob(path + '*.jpg'):
            im=Image.open(filename).resize((img_rows,img_cols)).convert('L')
            im=ImageOps.invert(im)      # Meaning of grey level is 255 (black) and 0 (white)
            image_list.append(np.asarray(im).astype('float32')/255)

    n = len(image_list)     # Total examplesvv
    if K.image_dim_ordering() == 'th':
        X = np.asarray(image_list).reshape(n,1,img_rows,img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = np.asarray(image_list).reshape(n,img_rows,img_cols,1)
        input_shape = (img_rows, img_cols, 1)
    Y = np.asarray(class_list)

    return X, Y, input_shape
