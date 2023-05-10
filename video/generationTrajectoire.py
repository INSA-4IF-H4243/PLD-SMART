import cv2
#!pip install .
import cv2
import numpy as np
import ffmpeg
from smart.processor import ImageProcessor
from smart.video import Video, Image
from pynput.keyboard import Key, Listener
from pynput import keyboard
import csv
import os
import math
import tkinter as tk
from PIL import Image
import numpy as np
import random

trajectoire = []
tab_point = []

def motion(event):
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))

def button1(event):
    """ Gestion de l'événement clic gauche sur la zone graphique """
    # MODE JEU
    # (X,Y) : position du pointeur de la souris
    x = event.x
    y = event.y
    trajectoire.append((x,y))

    can.create_rectangle(x-7, y-7, x+7, y+7)
    print("{}, {}".format(x,y))
    print(len(trajectoire))
    

def button2(event):
    """ retour en arriere """
    # MODE JEU
    trajectoire.pop()
    print("dernier point retiré")

def save_trajectoire(trajectoire, outPutPath) :
    if not os.path.exists(outPutPath):
        os.makedirs(outPutPath)
    
    with open(outPutPath+'/datasetTrajectoire','a') as f :
        writer = csv.writer(f)
        writer.writerow(trajectoire)
        f.close()


# videoResize=(600,300)
# cap = cv2.VideoCapture('datasetVideos\partie1.mp4')
# for i in range(300) :
#     ret1, frame1 = cap.read()
# frame1=cv2.resize(frame1,videoResize)

# np_array = np.array(frame1)
# imgpil = Image.fromarray(np_array) # Transformation du tableau en image PIL
# imgpil.save("resultat.jpg")

fen = tk.Tk()
fen.geometry("1000x500")
temp_img = tk.PhotoImage(file='resultat.png')
can = tk.Canvas(fen, width=600, height=300)
can.pack()

can.create_image(300, 150, image=temp_img)

fen.bind('<Button-1>', button1)
fen.bind('<Button-2>', button2)

fen.mainloop()
print("Choisis le type :")
type=int(input())

if type != 0 :

    ## creer des trajectoires similaires
    for i in range(100) :
        new_trajectoire = []
        for point in trajectoire :
            x = point[0] -5 + random.randrange(14)
            y = point[1] -5 + random.randrange(14)
            new_trajectoire.append((x,y))

        new_trajectoire.append(type)
        save_trajectoire(new_trajectoire,'trajectoire')

    trajectoire.append(type)
    save_trajectoire(trajectoire,'trajectoire')
    print('trajectoires enregistrées')

print("fin")