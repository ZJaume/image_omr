import cv2
import numpy as np
import sys
import re
import glob

dest = "data/HOMUS_filtered/"
source = "data/HOMUS_4Fold/"

def dilate(image, size, shape):
    if shape==1 :
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,size)
    elif shape==2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif shape==3:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,size)
    return cv2.dilate(image,kernel)

#
# Filter time classes to only
# 2/4, 3/4, 4/4, 6/8, cut and common time
#
def time_filter(paths):
    regex = r'([A-Za-z0-9 -_\/]+((2-2)|(3-8)|(9-8)|(12-8))-Time.*)'
    return [path for path in paths if not re.match(regex,elem)]

imgs = []
for i in range(1,5):
    imgs.extend(glob.glob('./data/HOMUS_4Fold/F{}/*'.format(i)))

for img in imgs:
    mat = cv2.imread(img,0)
    name = re.sub(r'\.pbm','',re.sub(r'\./data/HOMUS_4Fold/','',img))
    print(dest+name+'.png')
    cv2.imwrite(dest + name + '.png',mat,[9])
