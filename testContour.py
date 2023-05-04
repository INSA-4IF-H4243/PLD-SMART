import cv2
import numpy as np
import random

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rv_j1/cut2.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()
#tableau avec joueur 0 (en bas) et joueur 1 (en haut)
joueurs=[(700,250,100,200),(700,600,150,250)]
while cap.isOpened():
    # hauteur=len(frame1)
    # diff1 = cv2.absdiff(frame1, frame2)
    # diff2 = cv2.absdiff(frame2, frame3)
    # diff= cv2.absdiff(diff1, diff2)

    imgray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,200,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1,contours,-1,(0,0,255),4)

    tab_rec = []
    for contour in contours:
        rec_base = cv2.boundingRect(contour)
        (x, y, w, h) = rec_base
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow("feed", frame1)
    frame1 = frame2
    frame2=frame3
    ret, frame3 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()