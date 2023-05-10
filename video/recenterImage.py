import cv2
#!pip install .
import cv2
import numpy as np
from smart.processor import ImageProcessor
from smart.video import Video, Image
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
        img_out=imp.crop_silouhette(grayimg,50)
        img_out=imp.binary(img_out)
        images.append(img_out)


cv2.imshow('Gray image', img_out)
  
cv2.waitKey(0)
cv2.destroyAllWindows()
imp.save_ImageList(images, directory+'Recenter', True)
cv2.waitKey(0)
cv2.destroyAllWindows()