from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import glob
import random

black = 0
white = 255

homus_size = 40
border = 10

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
    for img in imgs:
        stave.paste(img,
                box=(box[0], box[1] + random.randint(-border*1,border*1)),
                mask=ImageOps.invert(img))
        box[0] += homus_size


imgs = []
for f in glob.glob("./data/A*.jpg"):
    imgs.append(Image.open(f))

stave = gen_stave(homus_size*len(imgs),int(homus_size*1.6))
put_symbols(stave, imgs)
stave.show(command='eog')
