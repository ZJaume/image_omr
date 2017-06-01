from PIL import Image, ImageDraw, ImageColor, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import glob
import random
import cv2
import sys

black = 0
white = 255

homus_size = 40
border = 10

nb_classes = 32

#
# Function that generates a music stave, lines-blanks ratio is 20% 80% by default
#
def gen_stave(width, heigth):
    blank_size = int(heigth*0.80)//4
    line_size = int(heigth*0.20)//5

    im = Image.new('L',(width,heigth + border*2), color=white)
    draw = ImageDraw.Draw(im)
    x = 0 + border
    num_lines = 0
    while x <= heigth + border*2 and num_lines < 5:
        draw.line([(0,x),(width,x)], fill=black, width=line_size)
        x += line_size + blank_size
        num_lines+=1
    #im.show()
    return im

#
# Given a list of images of symbols, paste it on the stave
#
def put_symbols(stave, symbols, offset=0):
    box = [0,int(homus_size*0.1) + border]
    for img in symbols:
        stave.paste(img,
                box=(box[0], box[1] + random.randint(-border*1,border*1)),
                mask=ImageOps.invert(img))
        box[0] += homus_size

#
# Generate a sequence of symbols and
# return the list of filepaths and list of corresponding labels
#
def gen_sequence(files, length):
    symbols = []
    labels = []
    for i in range(length):
        label = random.randint(0,nb_classes-1)
        f = random.randint(0,len(files[label])-1)
        symbols.append(Image.open(files[label][f]))
        labels.append(label)
    return symbols, labels

#
# Calculate the centroid of the blob in a note
#
def centroid(img):
    data = np.array(img.getdata()).reshape(homus_size,homus_size)
    cols = np.sum(data,axis=0)
    cols = outliers_filter(cols)
    rows = np.sum(data,axis=1)
    centroid = [np.argmin(cols),np.argmin(rows)]
    print("Centroid",centroid)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.point(centroid, fill=ImageColor.getrgb('red'))
    img.show()

    plt.plot(cols)
    plt.plot(rows)
    plt.ylabel("sum")
    plt.xlabel("pixels")
    plt.legend(['cols','rows'])
    plt.show()

def outliers_filter(array, m=1.75):
    std = np.std(array)
    mean = np.mean(array)
    for i in range(array.shape[0]):
        if not abs(array[i] - mean) < m * std:
            array[i] = mean
    return array

centroid(Image.open(sys.argv[1]))

# Create a list of lists containing filepaths of symbols
# divided by classes
imgs = []
for i in range(nb_classes):
    imgs.append(list(glob.glob('./data/HOMUS/train_{}/*'.format(i))))

labels = ""
for i in range(30):
    length = random.randint(1,8) #Number of characters in the sequence
    stave = gen_stave(homus_size*length,int(homus_size*1.6))
    symbols, label = gen_sequence(imgs, length)
    put_symbols(stave, symbols)
    for l in label:
        labels += str(l) + ' '
    labels += '\n'
    stave.save('./data/synth/{}.png'.format(i))

with open('./data/synth/labels.txt','w') as fp:
    fp.write(labels)
