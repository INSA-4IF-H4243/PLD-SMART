
import cv2
import numpy as np
import random

def cont(rec):
    contour=rec[2]*2+rec[3]*2
    return contour

def superposition(rec1, rec2):
    dx = min(rec1[0]+rec1[2], rec2[0]+rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[1]+rec1[3], rec2[1]+rec2[3]) - max(rec1[1], rec2[1])
    if (dx >= -60 and dy >= -60 and englobant) : 
        return True
    else :
        return False

def englobant(rec1, rec2):
    x1 = min(rec1[0], rec2[0])
    x2 = max(rec1[0]+rec1[2], rec2[0]+rec2[2])
    y1 = min(rec1[1], rec2[1])
    y2 = max(rec1[1]+rec1[3], rec2[1]+rec2[3])
    w = x2-x1
    h = y2-y1
    rec3 = (x1,y1,w,h)
    return rec3


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video_input2.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()
#tableau avec joueur 0 (en bas) et joueur 1 (en haut)
joueurs=[(0,0,0,0),(0,0,0,0)]
while cap.isOpened():
    diff1 = cv2.absdiff(frame1, frame2)
    diff2= cv2.absdiff(frame2, frame3)
    diff= cv2.absdiff(diff1, diff2)


    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tab_rec = []
    for contour in contours:
        rec_base = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 600:
            continue
        
        for rec in tab_rec:
            if superposition(rec_base, rec):
                rec_base = englobant(rec_base, rec)
                tab_rec.remove(rec)
        tab_rec.append(rec_base)
    #print("nb contour = ",len(tab_rec))

    # #retirer les petits après superpositon
    # for rec in tab_rec:
    #          print(rec)
    #          if cont(rec)<500:
    #              print(cont(rec))
    #              tab_rec.remove(rec)
                                     
    if(len(tab_rec)==2):
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

        #Jbas
        (x, y, w, h) = tab_rec[0]
        cv2.rectangle(frame1, (x-50, y-50), (x+150, y+250), (0, 255, 0), 2)
        #jhaut
        (x, y, w, h) = tab_rec[1]
        cv2.rectangle(frame1, (x-50, y-50), (x+150, y+150), (0, 255, 0), 2)
    else:
        for rec in tab_rec:
            (x, y, w, h) = rec
            cv2.rectangle(frame1, (x, y), (x+w, y+h),(0,0,255) , 2)
            #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    frame2=frame3
    ret, frame3 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()