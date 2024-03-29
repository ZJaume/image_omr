from PIL import Image, ImageDraw, ImageColor, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import glob
import re
import random
import cv2
import sys
import json

black = 0
white = 255

border = 30
sep = 15

nb_examples = 100000

dest = './data/synth/'
source = './data/HOMUS_filtered/'

#
# Function that generates a music stave, lines-blanks ratio is 15% 85% by default
#
def gen_stave(width, heigth):
    blank_size = int(heigth*0.85)//4
    line_size = int(heigth*0.15)//5

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
    box = [sep,int(stave.size[1]*0.05) + border]
    for img in symbols:
        stave.paste(img,
                box=(box[0], box[1] ),
                mask=ImageOps.invert(img))
        box[0] += img.size[0] + sep

#
# Generate a sequence of symbols and
# return the list of filepaths and list of corresponding labels
#
def gen_sequence(files, length, dic):
    symbols = []
    label = []
    size = sep    # Accumulate the width of every image
    for i in range(length):
        ran = random.randint(0,len(files)-1)
        label_i = parse_label(files[ran])
        if label_i not in dic:
            print("ERROR",files[ran])
            return  None, [files[ran]], None
        label_i = dic[label_i]
        symbols.append(ImageOps.invert(Image.open(files[ran])))
        label.append(label_i)
        size += symbols[-1].size[0] + sep
    return symbols, label, size

#
# Parse the label of an image from the filename
#
def parse_label(filename):
    return re.sub(r'\.png','',re.sub(r'\./data/HOMUS_filtered/F./W.*_.*_','',filename))

#
# Filter some unusual, useless or difficult classes
#
def class_filter(label):
    regex = r'((3-8)|(9-8)|(12-8)|(Sixty-Four)|(Thirty-Two)|(Natural)|(Double-Sharp)|(Sharp)|(Flat)|(Dot)|(Barline))+'
    return re.match(regex,label)

#
# Calculate where is  the head of the note
#
def centroid(img):
    print(img.size)
    print(img.mode)
    print(img.format)

    data = np.array(img.getdata(0)).reshape(img.size[0],img.size[1])
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

#centroid(ImageOps.invert(Image.open(sys.argv[1])))

# Create a list of filepaths of symbols
imgs = []
for i in range(1,5):
    imgs.extend(glob.glob(source + 'F{}/*'.format(i)))

# Get the class name from the filename
# and encode them in a dictionary json file
dictionary = {}
files = []
print(len(imgs))
for img in imgs:
    label = parse_label(img)
    if not class_filter(label):
        files.append(img)
        if label not in dictionary:
            dictionary[label] = len(dictionary)

with open(dest + 'dictionary.json', 'w') as f:
    json.dump(dictionary, f, sort_keys=True, indent=4)
print(dictionary)
print("---> Imported to json")

print("Generating {} examples...".format(nb_examples))
labels = ""
for i in range(nb_examples):
    length = random.randint(1,8) #Number of characters in the sequence
    symbols, label, size = gen_sequence(files, length, dictionary)
    if symbols is None:
        print(label in files)
        sys.exit(-1)
    stave = gen_stave(size,80)
    put_symbols(stave, symbols)
    for l in label:
        labels += str(l) + ' '
    labels += '\n'
    stave.save(dest + '{}.png'.format(i+1))

    if i%10000 == 0:
        print("{} examples generated...".format(i))
print("Done!")

with open(dest + 'labels.txt','w') as fp:
    fp.write(labels)
