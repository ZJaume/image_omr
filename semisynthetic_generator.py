from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

black = 0
white = 255

#
# Function that generates a music stave, lines-blanks ratio is 20% 80% by default
#
def gen_stave(width, heigth):
    blank_size = int(heigth*0.8)//4
    line_size = int(heigth*0.2)//5

    im = Image.new('L',(width,heigth), color=white)
    draw = ImageDraw.Draw(im)
    x = 0
    while x < heigth:
        draw.line([(0,x),(width,x)], fill=black, width=line_size)
        x += line_size + blank_size
    im.show()

gen_stave(300,120)
