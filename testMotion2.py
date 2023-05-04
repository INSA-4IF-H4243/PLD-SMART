
import cv2
import numpy as np
import random
def aire(rec):
    return rec[2]*rec[3]
def distCarre(rec1,rec2):
    return (rec1[1]-rec2[1])*(rec1[1]-rec2[1]) + (rec1[0]-rec2[0])*(rec1[0]-rec2[0])

def cont(rec):
    contour=rec[2]*2+rec[3]*2
    return contour

def superposition(rec1, rec2):
    dx = min(rec1[0]+rec1[2], rec2[0]+rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[1]+rec1[3], rec2[1]+rec2[3]) - max(rec1[1], rec2[1])
    if (dx >= -60 and dy >= -60) : 
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
joueurs=[(300,420,0,0),(300,420,0,0)]
while cap.isOpened():
    diff1 = cv2.absdiff(frame1, frame2)
    diff2= cv2.absdiff(frame2, frame3)
    diff= cv2.absdiff(diff1, diff2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #flow=cv2.calcOpticalFlowFarneback(gray, gray, None)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tab_rec = []
    for contour in contours:
        rec_base = cv2.boundingRect(contour)
        x = rec_base[0]
        y = rec_base[1]
        up_low_base = y < 420
        rec_base = cv2.boundingRect(contour)
        (x, y, w, h) = rec_base
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if cv2.contourArea(contour) < 100:
            continue
        #loop copie
        for rec in tab_rec[:]:
            
            
            up_low_rec = rec[1] < 420
            if superposition(rec_base, rec) and (up_low_base == up_low_rec or rec[3] < 70) :
                #if(aire(englobant(rec_base,rec))<aire(joueurs[0])+aire(joueurs[1])):
                rec_base = englobant(rec_base, rec)
                tab_rec.remove(rec)
        tab_rec.append(rec_base)
    #print("nb contour = ",len(tab_rec))

    #retirer les petits
    tab_rec = [rec for rec in tab_rec if ((not rec[2]<0))]
    #retirer les grands
    tab_rec = [rec for rec in tab_rec if ((not rec[3]>500) and not rec[2]>300)]
    #print("1: ",tab_rec)
    #tab_rec = [rec for rec in tab_rec if ((not rec[3]>200) and (not rec[2]<50))]
    #print("2: ",tab_rec)
    for rec in tab_rec:
        (x, y, w, h) = rec
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)


    if(len(tab_rec)==2):
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

    else:
        #cherche carré les plus proches des joueurs
        minJoueur0=(0,0,0,0)
        minJoueur1=(0,0,0,0)
        for rec in tab_rec:
            if distCarre(joueurs[0],rec)<distCarre(minJoueur1,joueurs[0]) and rec[1]<420:
                minJoueur0=rec
            if distCarre(joueurs[1],rec)<distCarre(minJoueur1,joueurs[1]) and rec[1]>420:
                minJoueur1=rec   

        joueurs[0] = minJoueur0
        joueurs[1] = minJoueur1

    #affichage des joueurs
    w1=max(joueurs[1][2]+50,150)
    h1=max(joueurs[1][3]+20,250)
    w0=max(joueurs[0][2]+50,150)
    h0=max(joueurs[0][3]+20,150)

    joueurs[0]=(joueurs[0][0]-50, joueurs[0][1]-20,w0,h0)
    joueurs[1]=(joueurs[1][0]-50, joueurs[1][1]-20,w1,h1)
    #Jbas
    (x, y, w, h) = joueurs[0]
    cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #jhaut
    (x, y, w, h) = joueurs[1]
    cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    frame2=frame3
    ret, frame3 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()