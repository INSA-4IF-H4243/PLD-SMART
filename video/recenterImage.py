import cv2
#!pip install .
import cv2
import numpy as np
from processor import ImageProcessor
from video import Video, Image
from pynput.keyboard import Key, Listener
from pynput import keyboard
from matplotlib import pyplot as plt

import os
 
directory = 'images'
images=[]
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)

        img = cv2.imread(f)
        imp=ImageProcessor()
        grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, final_img = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY)
        img_out=imp.crop_silouhette(final_img,50)
        _, final_img = cv2.threshold(img_out, 254, 255, cv2.THRESH_BINARY)
        images.append(final_img)

print(final_img)
cv2.imshow('Gray image', final_img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()
imp.save_ImageList(images, directory+'Recenter', True)
cv2.waitKey(0)
cv2.destroyAllWindows()