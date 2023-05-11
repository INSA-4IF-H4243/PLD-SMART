import copy
import cv2
#!pip install .
import cv2
import numpy as np
from smart.processor import ImageProcessor
from smart.processor import ImageProcessor, VideoProcessor
from smart.video import Video, Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder
import math
from smart.model import ModelJoueurClassique
from smart.model import ModelJoueurConvolution
########################PARAMETRES :

devMode=False#mode Développeur (=voir les tous les contours, filtres...)
affichage=True#est-ce qu'on veut afficher les resultats ou juste enregistrer ?
enregistrementImage=True#Est-ce qu'on veut enregistrer la sortie en image ou juste en tableau de 0 et de 1
PixelSizeOutput=50#taille de la sortie (=entree du machine learning)
videoPath='dataset/services.mp4'#chemin de la video
fpsOutput=7#FPS de la sortie
videoResize=(600,300)#taille pour resize de la video pour traitement (petite taille = plus rapide) 
cutFrameNB=15#nombre d'images pour un coups

y_pred_haut=4
y_pred_bas=4

#taille de lentree du machine learning = fpsOutput * [PixelSizeOutput * PixelSizeOutput] (20*20*20=8000 pixels noir ou blanc)
tableauSortieJHaut=[]
tableauSortieJBas=[]
tableau_position_balle = []
tableau_trajectoire_balle = []
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
    distance = math.sqrt((centre(rec1)[0]-centre(rec2)[0])*(centre(rec1)[0]-centre(rec2)[0]) + (centre(rec1)[1]-centre(rec2)[1])*(centre(rec1)[1]-centre(rec2)[1]))
    return distance

def distance_point(p1,p2):
    distance = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])
    return distance

def contour_taille(rec):
    contour=rec[2]*2+rec[3]*2
    return contour

def superposition(rec1, rec2):
    prox = -15
    dx = min(rec1[0]+rec1[2], rec2[0]+rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[1]+rec1[3], rec2[1]+rec2[3]) - max(rec1[1], rec2[1])
    if (dx >= prox and dy >= prox) or englobe(rec1, rec2): 
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
    if (h < 150 and w < 50) or englobe(rec1, rec2):
        rec3 = (x1,y1,w,h)
        return rec3
    return rec1

def englobe(rec1, rec2) : 
    if (rec1[0] <= (rec2[0]+10) and rec1[1] <= (rec2[1]+10) and (rec1[0]+rec1[2]) >= (rec2[0]+rec2[2]-10) and (rec1[1]+rec1[3]) >= (rec2[1]+rec2[3]-10) ) or (rec2[0] <= (rec1[0]+10) and rec2[1] <= (rec1[1]+10) and (rec2[0]+rec2[2]) >= (rec1[0]+rec1[2]-10) and (rec2[1]+rec2[3]) >= (rec1[1]+rec1[3]-10) ):
        return True
    return False


########REASEAU DE NEURONES:


input_shape_model=15*50*50
output_y=np.array([0,1,2,3]) #- 0: coup droit- 1: déplacement- 2: revers- 3: service
all_output_label = ['coup droit', 'deplacement', 'service', 'revers']

#JOUEUR BAS
model_bas = ModelJoueurClassique.load_model_from_path("saved_models/classic_model_1_joueur_bas.h5")
print(model_bas.summary_model)
#model_bas.load_model_from_path('JoueurBasTest.hdf5')


#JOUEUR HAUT
model_haut = ModelJoueurClassique.load_model_from_path("saved_models/classic_model_1_joueur_haut.h5")
print(model_haut.summary_model)        
#model_haut.load_model_from_path('JoueurHautTest.hdf5')

output_bas="nothing"
output_haut="nothing"
########TRAITEMENT DE LA VIDEO

#####LECTURE VIDEO
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
balle = (milieu_x-25,milieu_y,50,50)
pos_balle = centre(balle)
pos_precedent = pos_balle
balle_detecte = False
rayon_detection = 5
compteur_non_detection = 0
limite = 3

#####LECTURE IMAGE PAR IMAGE
nbFrame=0
print("...")
import cv2

import util
factor = 0.49
parameters = {
    "filter": {"iterations": 5, "shape": (10, 10)},  # brush size
    "substractor": {"history": 200, "threshold": 200},
}
subtractors = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
subtractor = util.subtractor(subtractors[2], parameters["substractor"])
while cap.isOpened() and ret3:#attention video qui s'arete au premier probleme dans la lecture a cause du resize
    ###AJUSTEMENT TAILLE
    # frame1=cv2.resize(frame1,videoResize)
    # frame2=cv2.resize(frame2,videoResize)

    transformations = []

    transformations.append(cv2.resize(frame3, (600, 300)))
    # cv2.imshow("frame", transformations[-1])

    transformations.append(transformations[-1][25:275, 100:600])
    # cv2.imshow("test", transformations[-1])

    transformations.append(cv2.cvtColor(transformations[-1], cv2.COLOR_BGR2GRAY))
    # cv2.imshow("gray", transformations[-1])

    transformations.append(subtractor.apply(transformations[-1]))
    cv2.imshow("mask", transformations[-1])

    transformations.append(
        util.filter(transformations[-1], "closing", parameters["filter"])
    )
    cv2.imshow("closing", transformations[-1])

    # transformations.append(util.filter(transformations[-1], "dilation", parameters["filter"]))
    # cv2.imshow("dilation", transformations[-1])

    # transformations.append(cv2.medianBlur(transformations[-1], 5))
    # cv2.imshow("blur", transformations[-1])

    ###RECHERCHE CONTOURS DES FORMES EN MOUVEMENT
    contours, hierarchy = cv2.findContours(
    transformations[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1:
            x, y, w, h = cv2.boundingRect(contour)
            if area > 20:
                cv2.rectangle(
                    transformations[0], (x + 100 -10, y + 25 -10), (x + 100 + w, y + 25 + h), (0, 0, 255), 1
                )  # players
            else:
                cv2.rectangle(
                    transformations[0], (x + 100 -10, y + 25 -10), (x + 100 + w, y + 25 + h), (0, 255, 0), 2
                )  # ball

    cv2.rectangle(transformations[0], (50, 25), (550, 275), (255, 255, 0), 1)
    


    ball_rec = []   #contient un tableau de contours qui va servire pour la balle
    ###PREMIERE DISCRIMINATION DES CONTOURS
    tab_rec = []    #contient les contours (contour=(x,y,w,h))
    for contour in contours:

        rec_base = cv2.boundingRect(contour)
        x = rec_base[0]
        y = rec_base[1]
        up_low_base = y < milieu_y*2/3
        (x, y, w, h) = rec_base
        new_rec = rec_base
        tab_rec.append(new_rec)
    ###AFFICHAGE DE TOUS LES CONTOURS
    
    if devMode:
        for rec in tab_rec:
            (x, y, w, h) = rec
            cv2.rectangle(transformations[0], (x, y), (x+w, y+h), (0, 0, 255), 2)
    print(len(tab_rec))
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
    
    (x, y, w, h) = joueurs[0] #Joueur 0 du haut
    #affichageJHaut=(max(0,x-decalageX), max(0,y-decalageY), w+2*decalageX, h+2*decalageY)
    affichageJHaut= (x+100 -10, y+25 -10, w + 10 ,h +10 )
    (x1, y1, w1, h1) = joueurs[1] #Joueur 1 du bas
    #affichageJBas=(x1-decalageX, y1-decalageY, w1+2*decalageX, h1+2*decalageY)
    affichageJBas=(x1+100 -10, y1+25 -10, w1 + 10, h1 + 10)

    ###DESSIN DU CONTOUR DES JOUEURS
    if(nbFrame%rapportFps<1):

        ###RECUPERATION SILOUHETTE 
        (x, y, w, h) = affichageJHaut
        (x1, y1, w1, h1) = affichageJBas
        try:
            crop_imgBas = imageProcessor.crop_frame_shadow_player(transformations[0], x1, x1+w1, y1, y1+h1)
            crop_imgHaut = imageProcessor.crop_frame_shadow_player(transformations[0], x, x+w, y, y+h)
            silouhetteHaut = imageProcessor.resize_img(crop_imgHaut,(PixelSizeOutput, PixelSizeOutput), interpolation=cv2.INTER_BITS)  
            silouhetteBas = imageProcessor.resize_img(crop_imgBas, (PixelSizeOutput, PixelSizeOutput), interpolation=cv2.INTER_BITS)
        except:
            silouhetteHaut = np.zeros((PixelSizeOutput,PixelSizeOutput))
            silouhetteBas = np.zeros((PixelSizeOutput,PixelSizeOutput))

        ###ENREGISTREMENT des silouhettes dans le TABLEAU
        tableauSortieJHaut.append(silouhetteHaut/255)
        tableauSortieJBas.append(silouhetteBas/255)
        
        ###PREDICTIONS

        
        #print(prected.shape)
        if(len(tableauSortieJBas)>15):
            seq_vid_bas=np.array(tableauSortieJBas[len(tableauSortieJBas)-cutFrameNB:len(tableauSortieJBas)]).reshape((1, 15*50*50))
            #(1, 50, 750, 3)
            output_bas = model_bas.predict_label(seq_vid_bas, all_output_label)[0]
            
   
        #print(prected.shape)
        if(len(tableauSortieJHaut)>15):
            seq_vid_haut=np.array(tableauSortieJHaut[len(tableauSortieJHaut)-cutFrameNB:len(tableauSortieJHaut)]).reshape((1, 15*50*50))
            output_haut = model_haut.predict_label(seq_vid_haut, all_output_label)[0]
            
        #print(" Joueur Haut: ", output_name[int(y_pred_haut)], (" Joueur Bas: ", output_name[int(y_pred_bas)]))
        
    ###AFFICHAGE 

    cv2.rectangle(transformations[0], (affichageJHaut[0], affichageJHaut[1]), (affichageJHaut[0]+affichageJHaut[2], affichageJHaut[1]+affichageJHaut[3]), (0, 200, 0), 2)
    cv2.rectangle(transformations[0], (affichageJBas[0], affichageJBas[1]), (affichageJBas[0]+affichageJBas[2], affichageJBas[1]+affichageJBas[3]), (0, 255, 0), 2)
    
    frame1=cv2.putText(transformations[0], output_haut, (affichageJHaut[0], affichageJHaut[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    frame1=cv2.putText(transformations[0], output_bas, (affichageJBas[0], affichageJBas[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

    cv2.imshow("feed", transformations[0])
    #if(devMode):cv2.imshow("feed2", dilated)

    cv2.imshow("JoueurHaut : ", silouhetteHaut)
    cv2.imshow("JoueurBas : ", silouhetteBas)

    ###CONTINUER LA LECTURE DE LA VIDEO
    frame1 = frame2
    frame2 = frame3
    ret3, frame3 = cap.read()
    nbFrame+=1

    if cv2.waitKey(40) == 27:
        break
    ret3, frame3 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()

###ENREGISTREMENT DONNEES:

# imageProcessor.save_ImageList(tableauSortieJHaut,outPutPathJHaut,enregistrementImage)
# imageProcessor.save_ImageList(tableauSortieJBas,outPutPathJBas,enregistrementImage)

print("fin")