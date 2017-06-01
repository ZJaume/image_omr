import cv2
import numpy as np

def dilate(image, size, shape):
    if shape==1 :
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,size)
    elif shape==2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif shape==3:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,size)
    return cv2.dilate(image,kernel)


