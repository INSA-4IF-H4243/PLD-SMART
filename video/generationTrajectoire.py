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

    can.create_rectangle(x-15, y-15, x+15, y+15)
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
    
    with open(outPutPath+'/datasetTrajectoire.csv','a') as f :
        writer = csv.writer(f)
        writer.writerow(trajectoire)
        f.close()


videoResize=(800,400)
cap = cv2.VideoCapture('dataset/partie2.mp4')

ret1, frame1 = cap.read()
frame1=cv2.resize(frame1,videoResize)

np_array = np.array(frame1)
imgpil = Image.fromarray(np_array) # Transformation du tableau en image PIL
imgpil.save("resultat.png")

fen = tk.Tk()
fen.geometry("1000x500")
temp_img = tk.PhotoImage(file='resultat.png')
can = tk.Canvas(fen, width=800, height=400)
can.pack()

can.create_image(400, 200, image=temp_img)

fen.bind('<Button-1>', button1)
fen.bind('<Button-2>', button2)

fen.mainloop()
print("Choisis le type :")
type=int(input())

if type != 0 :

    ## creer des trajectoires similaires
    for i in range(50) :
        new_trajectoire = []
        for point in trajectoire :
            x = point[0] -15 + random.randrange(30)
            y = point[1] -15 + random.randrange(30)
            new_trajectoire.append(x)
            new_trajectoire.append(y)

        new_trajectoire.append(type)
        save_trajectoire(new_trajectoire,'trajectoire')
    new_trajectoire = []
    for point in trajectoire :
            x = point[0]
            y = point[1]
            new_trajectoire.append(x)
            new_trajectoire.append(y)
    new_trajectoire.append(type)

    save_trajectoire(new_trajectoire,'trajectoire')
    print('trajectoires enregistrées')

print("fin")