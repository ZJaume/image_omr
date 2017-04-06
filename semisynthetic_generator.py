from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import glob

black = 0
white = 255

homus_size = 40

#
# Function that generates a music stave, lines-blanks ratio is 20% 80% by default
#
def gen_stave(width, heigth):
    blank_size = int(heigth*0.80)//4
    line_size = int(heigth*0.20)//5

    im = Image.new('L',(width,heigth), color=white)
    draw = ImageDraw.Draw(im)
    x = 0
    while x < heigth:
        draw.line([(0,x),(width,x)], fill=black, width=line_size)
        x += line_size + blank_size
    im.show()
    return im

#
# Given a list of images of symbols, paste it on the stave
#
def put_symbols(stave, symbols, offset=0):
    box = [0,int(homus_size*0.1)]
    for img in imgs:
        stave.paste(img, box=tuple(box), mask=ImageOps.invert(img))
        box[0] += homus_size


imgs = []
for f in glob.glob("./data/A*.jpg"):
    imgs.append(Image.open(f))

stave = gen_stave(homus_size*len(imgs),int(homus_size*1.2))
put_symbols(stave, imgs)
stave.show()
