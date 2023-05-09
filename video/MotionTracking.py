import cv2
import numpy as np
import random
#!pip install .
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ffmpeg
ffmpeg.__path__
from smart.processor import ImageProcessor, VideoProcessor
from smart.video import Video, Image

########################PARAMETRES :

devMode=False#mode Développeur (=voir les tous les contours, filtres...)
affichage=True#est-ce qu'on veut afficher les resultats ou juste enregistrer ?
enregistrementImage=True#Est-ce qu'on veut enregistrer la sortie en image ou juste en tableau de 0 et de 1
PixelSizeOutput=20#taille de la sortie (=entree du machine learning)
videoPath='datasetVideosCourtes/v.mp4'#chemin de la video
outPutPathJHaut='img/cd_j133/cut/jHaut'#chemin d'enregistrement de la silouhette du Joueur 1
outPutPathJBas='img/cd_j133/cut/jBas'#chemin d'enregistrement de la silouhette du Joueur 2
fpsOutput=20#FPS de la sortie
videoResize=(600,300)#taille pour resize de la video pour traitement (petite taille = plus rapide) 

#taille de lentree du machine learning = fpsOutput * [PixelSizeOutput * PixelSizeOutput] (20*20*20=8000 pixels noir ou blanc)
tableauSortieJHaut=[]
tableauSortieJBas=[]
print(tableauSortieJHaut)

########################METHODES TRAITEMENT CONTOURS :

def aire(rec):
    return rec[2]*rec[3]

def centre(rec):
    #centre = (x,y)
    centre=(rec[0]+rec[2]/2,rec[1]+rec[3]/2)
    return centre

def similarite(rec1,rec2):
     distance = (centre(rec1)[0]-centre(rec2)[0])*(centre(rec1)[0]-centre(rec2)[0]) + (centre(rec1)[1]-centre(rec2)[1])*(centre(rec1)[1]-centre(rec2)[1])
     #differenceAire = abs((rec2[2]*rec2[3])-(rec1[2]*rec1[3]))
     return distance

def distance2(rec1,rec2):
    distance = (centre(rec1)[0]-centre(rec2)[0])*(centre(rec1)[0]-centre(rec2)[0]) + (centre(rec1)[1]-centre(rec2)[1])*(centre(rec1)[1]-centre(rec2)[1])
    return distance

def contour_taille(rec):
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
    if h < 150 and w < 50:
        rec3 = (x1,y1,w,h)
        return rec3
    return rec1

########################

########TRAITEMENT DE LA VIDEO

#####LECTURE
#cap = cv2.VideoCapture('rv_j1/cut6.mp4')
cap = cv2.VideoCapture(videoPath)
fps = cap.get(cv2.CAP_PROP_FPS)#FPS de la video d'entree
rapportFps=fps/fpsOutput
imageProcessor = ImageProcessor()

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()
ret3, frame3 = cap.read()

#####AJUSTEMENT TAILLE
frame1=cv2.resize(frame1,videoResize)
milieu_y=int(len(frame1)/2)
milieu_x=int(len(frame1[0])/2)

#####INIT CONTOURS JOUEURS AU MILIEU DU TERRAIN (joeur 0 = joueur du haut, joueur 1 = joueur du bas)
joueurs=[(milieu_x-25,milieu_y-75,50,50),(milieu_x-25,milieu_y+75,50,50)]

#####LECTURE IMAGE PAR IMAGE
nbFrame=0
print("...")
while cap.isOpened() and ret3:#attention video qui s'arete au premier probleme dans la lecture a cause du resize
    #print(frame3)
    ###AJUSTEMENT TAILLE
    frame1=cv2.resize(frame1,videoResize)
    frame2=cv2.resize(frame2,videoResize)
    frame3=cv2.resize(frame3,videoResize)

    ###DIFFERENCE IMAGES POUR VOIR LES PIXELS EN MOUVEMENTS
    diff1 = cv2.absdiff(frame1, frame2)
    diff2 = cv2.absdiff(frame2, frame3)
    diff= cv2.absdiff(diff1, diff2)

    ###TRAITEMENT ET BINARISATION
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #flow=cv2.calcOpticalFlowFarneback(gray, gray, None)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=3)

    ###RECHERCHE CONTOURS DES FORMES EN MOUVEMENT
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    ###PREMIERE DISCRIMINATION DES CONTOURS
    tab_rec = []    #contient les contours (contour=(x,y,w,h))
    for contour in contours:

        rec_base = cv2.boundingRect(contour)
        x = rec_base[0]
        y = rec_base[1]
        up_low_base = y < milieu_y*2/3
        (x, y, w, h) = rec_base
        new_rec = rec_base
        if devMode:cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)

        #élimination des contours dont la forme est clairement differente d'un tennisman
        if( contour_taille(rec_base)<100 or contour_taille(rec_base)>1000):continue
        if(rec_base[2]/rec_base[3]>4 or rec_base[3]/rec_base[2]>4):continue

        #fusion des contours proches
        for rec in tab_rec[:]:         
            up_low_rec = rec[1] < milieu_y*2/3
            if superposition(rec_base, rec) and (up_low_base == up_low_rec or rec[3] < 30 ) :
                new_rec = englobant(rec_base, rec)
                if new_rec != rec_base:
                    tab_rec.remove(rec)
        tab_rec.append(new_rec)

    ###AFFICHAGE DE TOUS LES CONTOURS
    if devMode:
        for rec in tab_rec:
            (x, y, w, h) = rec
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    ###CHOIX FINAL DES DEUX CONTOURS DES DEUX JOUEURS
    if(len(tab_rec)==2):       #Si à cette étape il n'y a que 2 contours, ce sont les bons
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

    else:                       #Sinon on prend les contours les plus proches des anciens contours identifiés comme ceux des joueurs (=tracking)
        minJoueur0=(milieu_x-25,milieu_y-75,50,50)
        minJoueur1=(milieu_x-25,milieu_y+75,50,50)
        b0 = 0
        b1 = 0
        for rec in tab_rec: 
            #joueur du haut          
            if distance2(joueurs[0],rec) < distance2(joueurs[0],minJoueur0) and (centre(rec)[1]<milieu_y):
                #if distance2(joueurs[0],rec) < 5*5:
                    if devMode:print("joueur0")
                    if devMode:print(similarite(joueurs[0],rec))
                    minJoueur0=rec
                    b0 = 1
            #joueur du bas
            elif distance2(joueurs[1],rec) < distance2(minJoueur1,joueurs[1]) and (centre(rec)[1]>milieu_y):
                #if distance2(joueurs[1],rec) < 5*5:
                    minJoueur1=rec  
                    b1 = 1 
        if b0:
            joueurs[0] = minJoueur0
        if b1:
            joueurs[1] = minJoueur1

    ###DESSIN DU CONTOUR DES JOUEURS
    if(nbFrame%rapportFps<1):

        ###CREATION CONTOUR AVEC DECALAGE
        decalageX = int(milieu_x/30)
        decalageY = int(milieu_y/15)
        
        (x, y, w, h) = joueurs[0] #Joueur 0 du haut
        affichageJHaut=(max(0,x-decalageX), max(0,y-decalageY), w+2*decalageX, h+2*decalageY)
        cv2.rectangle(frame1, (affichageJHaut[0], affichageJHaut[1]), (affichageJHaut[0]+affichageJHaut[2], affichageJHaut[1]+affichageJHaut[3]), (0, 200, 0), 2)
        (x1, y1, w1, h1) = joueurs[1] #Joueur 1 du bas
        affichageJBas=(x1-decalageX, y1-decalageY, w1+2*decalageX, h1+2*decalageY)
        cv2.rectangle(frame1, (affichageJBas[0], affichageJBas[1]), (affichageJBas[0]+affichageJBas[2], affichageJBas[1]+affichageJBas[3]), (0, 255, 0), 2)

        ###RECUPERATION SILOUHETTE 
        (x, y, w, h) = affichageJHaut
        (x1, y1, w1, h1) = affichageJBas

        crop_imgBas = imageProcessor.crop_frame_shadow_player(frame1, x1, x1+w1, y1, y1+h1)
        silouhetteBas=imageProcessor.resize_img(crop_imgBas, (PixelSizeOutput, PixelSizeOutput))

        crop_imgHaut = imageProcessor.crop_frame_shadow_player(frame1, x, x+w, y, y+h)
        silouhetteHaut = imageProcessor.resize_img(crop_imgHaut,(PixelSizeOutput, PixelSizeOutput))

        ###AFFICHAGE 
        if(affichage):

            cv2.imshow("feed", frame1)
            if(devMode):cv2.imshow("feed2", dilated)

            cv2.imshow("JoueurHaut", silouhetteHaut)
            cv2.imshow("JoueurBas", silouhetteBas)

        ###ENREGISTREMENT des silouhettes dans le TABLEAU
        tableauSortieJHaut.append(silouhetteHaut/255)
        tableauSortieJBas.append(silouhetteBas/255)

    ###CONTINUER LA LECTURE DE LA VIDEO
    frame1 = frame2
    frame2 = frame3
    ret3, frame3 = cap.read()
    nbFrame+=1

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()

###ENREGISTREMENT DONNEES:

imageProcessor.save_ImageList(tableauSortieJHaut,outPutPathJHaut,enregistrementImage)
imageProcessor.save_ImageList(tableauSortieJBas,outPutPathJBas,enregistrementImage)

print("fin")