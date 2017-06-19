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

imgs = []
for i in range(1,5):
    imgs.extend(glob.glob('./data/HOMUS_4Fold/F{}/*'.format(i)))

for img in imgs:
    mat = dilate(cv2.imread(img,0), (3,3), 2)
    name = re.sub(r'\.pbm','',re.sub(r'\./data/HOMUS_4Fold/','',img))
    cv2.imwrite(dest + name + '.png',mat,[9])
