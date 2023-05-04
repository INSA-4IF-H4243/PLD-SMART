import cv2
import numpy as np
import random
def aire(rec):
    return rec[2]*rec[3]

def centre(rec):
    #centre = (x,y)
    centre=(rec[0]+rec[2]/2,rec[1]+rec[3]/2)
    return centre

def similarite(rec1,rec2):
    distance = (centre(rec1)[0]-centre(rec2)[0])*(centre(rec1)[0]-centre(rec2)[0]) + (centre(rec1)[1]-centre(rec2)[1])*(centre(rec1)[1]-centre(rec2)[1])
    differenceAire = abs((rec2[2]*rec2[3])-(rec1[2]*rec1[3]))
    return distance

def distance2(rec1,rec2):
    distance = (centre(rec1)[0]-centre(rec2)[0])*(centre(rec1)[0]-centre(rec2)[0]) + (centre(rec1)[1]-centre(rec2)[1])*(centre(rec1)[1]-centre(rec2)[1])
    return distance

def cont(rec):
    contour=rec[2]*2+rec[3]*2
    return contour

def superposition(rec1, rec2):
    prox = -15
    dx = min(rec1[0]+rec1[2], rec2[0]+rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[1]+rec1[3], rec2[1]+rec2[3]) - max(rec1[1], rec2[1])
    if (dx >= prox and dy >= prox) : 
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
    if h < 300 and w < 200:
        rec3 = (x1,y1,w,h)
        return rec3
    return rec1


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rv_j1/cut2.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()
milieu_y=len(frame1)/2
milieu_x=len(frame1[0])/2
print("milieuy = ",milieu_y)
print("milieux = ",milieu_x)
#tableau avec joueur 0 (en bas) et joueur 1 (en haut)
joueurs=[(milieu_x,milieu_y,100,200),(milieu_x,milieu_y,150,250)]
while cap.isOpened():

    devMode=True

    diff1 = cv2.absdiff(frame1, frame2)
    diff2 = cv2.absdiff(frame2, frame3)
    diff= cv2.absdiff(diff1, diff2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #flow=cv2.calcOpticalFlowFarneback(gray, gray, None)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    tab_rec = []
    for contour in contours:
        rec_base = cv2.boundingRect(contour)
        x = rec_base[0]
        y = rec_base[1]
        up_low_base = y < 420
        (x, y, w, h) = rec_base
        new_rec = rec_base
        if devMode:cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)
        #loop copie

        #traitement des contours dont la forme est clairement differente d'un tennisman
        if(cont(rec_base)<100 or cont(rec_base)>1000):continue
        if(rec_base[2]/rec_base[3]>5 or rec_base[3]/rec_base[2]>5):continue
        for rec in tab_rec[:]:         
            up_low_rec = rec[1] < milieu_y
            if superposition(rec_base, rec) and (up_low_base == up_low_rec or rec[3] < 70) :
                new_rec = englobant(rec_base, rec)
                if new_rec != rec_base:
                    tab_rec.remove(rec)
        tab_rec.append(new_rec)
    #print("nb contour = ",len(tab_rec))
    #retirer les petits
    #tab_rec = [rec for rec in tab_rec if ((not rec[2]<20) and not rec[3]<40)]
    #retirer les grands
    #tab_rec = [rec for rec in tab_rec if ((not rec[3]>350) and not rec[2]>200)]
    #print("1: ",tab_rec)
    #tab_rec = [rec for rec in tab_rec if ((not rec[3]>200) and (not rec[2]<50))]
    #print("2: ",tab_rec)

    for rec in tab_rec:
        (x, y, w, h) = rec
        if devMode:cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if(len(tab_rec)==2):
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

    else:
        #cherche carré les plus proches des joueurs
        minJoueur0=(10000000,10000000,10000000,10000000)
        minJoueur1=(10000000,10000000,10000000,10000000)
        b0 = 0
        b1 = 0
        for rec in tab_rec:           
            if distance2(joueurs[0],rec) < distance2(joueurs[0],minJoueur0) and centre(rec)[1]<milieu_y:
                if devMode:print("joueur0")
                if devMode:print(similarite(joueurs[0],rec))
                minJoueur0=rec
                b0 = 1
#and similarite(joueurs[1],rec) < 2000 and rec[1]>420:
            elif distance2(joueurs[1],rec) < distance2(minJoueur1,joueurs[1]) and centre(rec)[1]>milieu_y:
                minJoueur1=rec  
                b1 = 1 
        if b0:
            joueurs[0] = minJoueur0
        if b1:
            joueurs[1] = minJoueur1

    #affichage des joueurs
    decalageX = 0
    decalageY = 0

    # w1=max(joueurs[1][2]+ decalageX + 20,150)
    # h1=max(joueurs[1][3]+ decalageY + 20,250)
    # w0=max(joueurs[0][2]+ decalageX + 20,150)
    # h0=max(joueurs[0][3]+ decalageY + 20,150)

    # joueurs[0]=(joueurs[0][0]-50, joueurs[0][1]-20,w0,h0)
    # joueurs[1]=(joueurs[1][0]-50, joueurs[1][1]-20,w1,h1)

    #Jhaut
    (x, y, w0, h0) = joueurs[0]
    w=max(w0,25)
    h=max(h0,40)
    cv2.rectangle(frame1, (x-decalageX, y-decalageY), (x+w+decalageX, y+h+decalageY), (0, 255, 0), 2)
    #jbas
    (x, y, w1, h1) = joueurs[1]
    w=max(w1,75)
    h=max(h1,120)
    cv2.rectangle(frame1, (x-decalageX, y-decalageY), (x+w+decalageX, y+h+decalageY), (0, 255, 0), 2)
    cv2.imshow("feed", frame1)
    cv2.imshow("feed2", dilated)
    #cv2.imshow("feed2", dilated)
    frame1 = frame2
    frame2=frame3
    ret, frame3 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()