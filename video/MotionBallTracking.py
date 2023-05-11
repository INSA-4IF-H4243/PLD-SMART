import cv2
#!pip install .
import cv2
import numpy as np
import ffmpeg
from smart.processor import ImageProcessor
from smart.video import Video, Image
import math
import copy
import util

########################PARAMETRES :
HAUT_VERS_BAS = 1
BAS_VERS_HAUT = 0
PAS_MOUVEMENT_BALLE = -1

devMode=False#mode Développeur (=voir les tous les contours, filtres...)
affichage=True#est-ce qu'on veut afficher les resultats ou juste enregistrer ?
enregistrementImage=True#Est-ce qu'on veut enregistrer la sortie en image ou juste en tableau de 0 et de 1
PixelSizeOutput=20#taille de la sortie (=entree du machine learning)
videoPath='datasetVideos/partie1.mp4'#chemin de la video
outPutPathJHaut='img/cd_j133/cut/jHaut'#chemin d'enregistrement de la silouhette du Joueur 1
outPutPathJBas='img/cd_j133/cut/jBas'#chemin d'enregistrement de la silouhette du Joueur 2
fpsOutput=20#FPS de la sortie
videoResize=(600,300)#taille pour resize de la video pour traitement (petite taille = plus rapide) 

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

########################

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

############################################################

VIDEO = "video_input2"
VIDEO_INPUT = f"inputs/{VIDEO}.mp4"
VIDEO_OUTPUT = f"outputs/{VIDEO}.avi"

factor = 0.49

parameters = {"substractor": {"history": 200, "threshold": 200}}

subtractors = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
subtractor = util.subtractor(subtractors[2], parameters["substractor"])

capture = cv2.VideoCapture(videoPath)
exist, frame = capture.read()

ymin = 30
ymax = 370
xmin = 80
xmax = 720

while exist:
    transformations = []

    transformations.append(cv2.resize(frame, (800,400)))
    # cv2.imshow("frame", transformations[-1])

    transformations.append(transformations[-1][ymin:ymax, xmin:xmax])
    # cv2.imshow("test", transformations[-1])

    transformations.append(cv2.cvtColor(transformations[-1], cv2.COLOR_BGR2GRAY))
    # cv2.imshow("gray", transformations[-1])

    transformations.append(subtractor.apply(transformations[-1]))
    cv2.imshow("mask", transformations[-1])

    # transformations.append(
    #     util.filter(transformations[-1], "opening", parameters["filter"])
    # )

    transformations.append(util.filter(transformations[-1], "closing"))
    cv2.imshow("closing", transformations[-1])
    transformations.append(util.filter(transformations[-1], "dilation"))  


    cv2.imshow("final", transformations[-1])

    # transformations.append(util.filter(transformations[-1], "dilation", parameters["filter"]))
    # cv2.imshow("dilation", transformations[-1])

    # transformations.append(cv2.medianBlur(transformations[-1], 5))
    # cv2.imshow("blur", transformations[-1])

    contours, hierarchy = cv2.findContours(
        transformations[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tab_rec = []
    ball_rec =[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1:
            x, y, w, h = cv2.boundingRect(contour)
            if area > 2000 : continue
            if area > 300:
                tab_rec.append((x, y, w, h))
            else:
                ball_rec.append((x, y, w, h))

    if(len(tab_rec)==2):       #Si à cette étape il n'y a que 2 contours, ce sont les bons
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

    else:                       #Sinon on prend les contours les plus proches des anciens contours identifiés comme ceux des joueurs (=tracking)
        minJoueur0=(10000,10000,10000,10000)
        minJoueur1=(10000,10000,10000,10000)
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

    minBalle = (1000000, 1000000, 1000000, 1000000, 1000000)
    bBalle = 0
    compteur = 0

    if balle_detecte:
        for rec in ball_rec :
            if distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 20 and distance2(balle,rec) > 1 :
                minBalle=rec  
                bBalle = 1
        if bBalle : balle = minBalle
        else : balle_detecte = False

    b = False
    if not balle_detecte:
        if compteur_non_detection < limite :
            for rec in ball_rec :
                if (distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 20+rayon_detection*compteur_non_detection) :
                    minBalle=rec  
                    bBalle = 1
                    b = True
        else :
            for rec in ball_rec :
                if not b and ((distance2(joueurs[0],rec) < distance2(joueurs[0],minBalle) and distance2(joueurs[0],rec) < 100 and joueurs[0][1]-20 < rec[1]) or (distance2(joueurs[1],rec) < distance2(joueurs[1],minBalle) and distance2(joueurs[1],rec) < 100 and joueurs[1][1]+joueurs[1][3]+20 > rec[1])) :
                    minBalle=rec  
                    bBalle = 1

    if bBalle : 
        balle = minBalle
        balle_detecte = True
        compteur_non_detection = 0
        pos_balle = centre(balle)
        tableau_trajectoire_balle.append(pos_balle)

    else :
        tableau_trajectoire_balle.append((-1,-1))
        compteur_non_detection+=1

    if len(tableau_trajectoire_balle) > 45 :
        tableau_trajectoire_balle.pop(0)

    if len(tableau_trajectoire_balle) == 45 :

        tableau_position_balle = copy.deepcopy(tableau_trajectoire_balle)
        no_pos = 0
        no_debut = False

        for i in range(len(tableau_trajectoire_balle)) :
            
            if tableau_trajectoire_balle[i] == (-1,-1) :
                no_pos+=1

                if i == 0 :
                    no_debut = True

                if no_pos == 1 and not no_debut:
                        av_derniere_pos_balle = tableau_trajectoire_balle[i-1]

                if i == len(tableau_trajectoire_balle)-1 and not no_debut:
                    for j in range(0,no_pos+1) :
                        tableau_position_balle[i-j] = derniere_pos_balle
            else :
                derniere_pos_balle = tableau_trajectoire_balle[i]
                if no_pos > 0 :
                    if no_debut :
                        for j in range(1,no_pos+1) :
                            tableau_position_balle[i-j] = derniere_pos_balle
                        no_debut = False
                    else :
                        for j in range(1,no_pos+1) :
                            x = av_derniere_pos_balle[0] + int (((derniere_pos_balle[0] - av_derniere_pos_balle[0])/no_pos)*j)
                            y = av_derniere_pos_balle[1] + int (((derniere_pos_balle[1] - av_derniere_pos_balle[1])/no_pos)*j)
                            tableau_position_balle[i-j] = (x,y)
                no_pos = 0

    (x, y, w, h) = balle

    for joueur in joueurs :
        cv2.rectangle(
                    transformations[0], (joueur[0] + xmin - 10, joueur[1] + ymin - 10), (joueur[0] + xmin + joueur[2], joueur[1] + ymin + joueur[3]), (0, 0, 255), 2
                )  # players

    if balle_detecte : 
        cv2.rectangle(
                    transformations[0], (balle[0] + xmin - 10, balle[1] + ymin - 10), (balle[0] + xmin + balle[2], balle[1] + ymin + balle[3]), (0, 255, 0), 2
                )  # ball
        

    cv2.rectangle(transformations[0], (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imshow("frame", transformations[0])

    # codec = cv2.VideoWriter_fourcc(*'XVID')
    # height = transformations[0].shape[0]
    # width = transformations[0].shape[1]
    # writer = cv2.VideoWriter(VIDEO_OUTPUT, codec, 25, (width, height), False)

    exist, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

############################################################


#####LECTURE IMAGE PAR IMAGE
# nbFrame=0
# print("...")
# while cap.isOpened() and ret3:#attention video qui s'arete au premier probleme dans la lecture a cause du resize
#     ###AJUSTEMENT TAILLE
#     frame1=cv2.resize(frame1,videoResize)
#     frame2=cv2.resize(frame2,videoResize)
#     frame3=cv2.resize(frame3,videoResize)

#     ###DIFFERENCE IMAGES POUR VOIR LES PIXELS EN MOUVEMENTS
#     diff1 = cv2.absdiff(frame1, frame2)
#     diff2 = cv2.absdiff(frame2, frame3)
#     diff= cv2.absdiff(diff1, diff2)

#     ###TRAITEMENT ET BINARISATION
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     #flow=cv2.calcOpticalFlowFarneback(gray, gray, None)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_OTSU)
#     dilated = cv2.dilate(thresh, None, iterations=3)

#     ###RECHERCHE CONTOURS DES FORMES EN MOUVEMENT
#     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     ###PREMIERE DISCRIMINATION DES CONTOURS
#     tab_rec = []    #contient les contours (contour=(x,y,w,h))
#     ball_rec = []   #contient un tableau de contours qui va servire pour la balle
#     for contour in contours:

#         rec_base = cv2.boundingRect(contour)
#         x = rec_base[0]
#         y = rec_base[1]
#         up_low_base = y < milieu_y*2/3
#         (x, y, w, h) = rec_base
#         new_rec = rec_base

#         if devMode:cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)

#         #élimination des contours dont la forme est clairement differente d'un tennisman
        
#         if(contour_taille(rec_base)>1000):continue

#         if(rec_base[2]/rec_base[3]>4 or rec_base[3]/rec_base[2]>4):continue

#         #fusion des contours proches
#         if(contour_taille(rec_base)<100):
#             for rec in ball_rec[:]:         
#                 if superposition(rec_base, rec):
#                     new_rec = englobant(rec_base, rec)
#                     if new_rec != rec_base:
#                         ball_rec.remove(rec)
#             if contour_taille(new_rec) > 10 : ball_rec.append(new_rec)

#         else :
#             for rec in tab_rec[:]:         
#                 up_low_rec = rec[1] < milieu_y*2/3
#                 if superposition(rec_base, rec) and (up_low_base == up_low_rec or rec[3] < 30 ) :
#                     new_rec = englobant(rec_base, rec)
#                     if new_rec != rec_base:
#                         tab_rec.remove(rec)
#                         if rec in ball_rec : ball_rec.remove(rec)
#             tab_rec.append(new_rec)
#             if new_rec in ball_rec : 
#                 if new_rec[2]>30 or new_rec[3]>75 : ball_rec.remove(new_rec)
#                 else : ball_rec.append(new_rec)

#     ###AFFICHAGE DE TOUS LES CONTOURS
#     if devMode:
#         for rec in tab_rec:
#             (x, y, w, h) = rec
#             cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

#     ###CHOIX FINAL DES DEUX CONTOURS DES DEUX JOUEURS
#     if(len(tab_rec)==2):       #Si à cette étape il n'y a que 2 contours, ce sont les bons
#         if((tab_rec[0])[1]<(tab_rec[1])[1]):
#             joueurs[0]=tab_rec[0]
#             joueurs[1]=tab_rec[1]
#         else:
#             joueurs[0]=tab_rec[1]
#             joueurs[1]=tab_rec[0]

#     else:                       #Sinon on prend les contours les plus proches des anciens contours identifiés comme ceux des joueurs (=tracking)
#         minJoueur0=(milieu_x-25,milieu_y-75,50,50)
#         minJoueur1=(milieu_x-25,milieu_y+75,50,50)
#         b0 = 0
#         b1 = 0
#         for rec in tab_rec: 
#             #joueur du haut          
#             if distance2(joueurs[0],rec) < distance2(joueurs[0],minJoueur0) and (centre(rec)[1]<milieu_y):
#                 #if distance2(joueurs[0],rec) < 5*5:
#                     if devMode:print("joueur0")
#                     if devMode:print(similarite(joueurs[0],rec))
#                     minJoueur0=rec
#                     b0 = 1
#             #joueur du bas
#             elif distance2(joueurs[1],rec) < distance2(minJoueur1,joueurs[1]) and (centre(rec)[1]>milieu_y):
#                 #if distance2(joueurs[1],rec) < 5*5:
#                     minJoueur1=rec  
#                     b1 = 1 
#         if b0:
#             joueurs[0] = minJoueur0
#         if b1:
#             joueurs[1] = minJoueur1

#     for rec1 in joueurs:
#         for rec2 in ball_rec[:] :
#             if englobe(rec1, rec2): 
#                 ball_rec.remove(rec2)

#     # Trouver la balle

#     minBalle = (1000000, 1000000, 1000000, 1000000, 1000000)
#     bBalle = 0
#     compteur = 0

#     if balle_detecte:
#         for rec in ball_rec :
#             if distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 20 and distance2(balle,rec) > 1 :
#                 minBalle=rec  
#                 bBalle = 1
#         if bBalle : balle = minBalle
#         else : balle_detecte = False

#     b = False
#     if not balle_detecte:
#         if compteur_non_detection < limite :
#             for rec in ball_rec :
#                 if (distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 20+rayon_detection*compteur_non_detection) :
#                     minBalle=rec  
#                     bBalle = 1
#                     b = True
#         else :
#             for rec in ball_rec :
#                 if not b and ((distance2(joueurs[0],rec) < distance2(joueurs[0],minBalle) and distance2(joueurs[0],rec) < 100 and joueurs[0][1]-20 < rec[1]) or (distance2(joueurs[1],rec) < distance2(joueurs[1],minBalle) and distance2(joueurs[1],rec) < 100 and joueurs[1][1]+joueurs[1][3]+20 > rec[1])) :
#                     minBalle=rec  
#                     bBalle = 1

#     if bBalle : 
#         balle = minBalle
#         balle_detecte = True
#         compteur_non_detection = 0
#         pos_balle = centre(balle)
#         tableau_trajectoire_balle.append(pos_balle)

#     else :
#         tableau_trajectoire_balle.append((-1,-1))
#         compteur_non_detection+=1

#     if len(tableau_trajectoire_balle) > 45 :
#         tableau_trajectoire_balle.pop(0)

#     if len(tableau_trajectoire_balle) == 45 :

#         tableau_position_balle = copy.deepcopy(tableau_trajectoire_balle)
#         no_pos = 0
#         no_debut = False

#         for i in range(len(tableau_trajectoire_balle)) :
            
#             if tableau_trajectoire_balle[i] == (-1,-1) :
#                 no_pos+=1

#                 if i == 0 :
#                     no_debut = True

#                 if no_pos == 1 and not no_debut:
#                         av_derniere_pos_balle = tableau_trajectoire_balle[i-1]

#                 if i == len(tableau_trajectoire_balle)-1 :
#                     for j in range(0,no_pos+1) :
#                         tableau_position_balle[i-j] = derniere_pos_balle
#             else :
#                 derniere_pos_balle = tableau_trajectoire_balle[i]
#                 if no_pos > 0 :
#                     if no_debut :
#                         for j in range(1,no_pos+1) :
#                             tableau_position_balle[i-j] = derniere_pos_balle
#                         no_debut = False
#                     else :
#                         for j in range(1,no_pos+1) :
#                             x = av_derniere_pos_balle[0] + int (((derniere_pos_balle[0] - av_derniere_pos_balle[0])/no_pos)*j)
#                             y = av_derniere_pos_balle[1] + int (((derniere_pos_balle[1] - av_derniere_pos_balle[1])/no_pos)*j)
#                             tableau_position_balle[i-j] = (x,y)
#                 no_pos = 0

        
            
    
#     (x, y, w, h) = balle
#     if balle_detecte : cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 255, 0), 2)

#     ###DESSIN DU CONTOUR DES JOUEURS

#     ###CREATION CONTOUR AVEC DECALAGE
#     decalageX = int(milieu_x/30)
#     decalageY = int(milieu_y/15)
    
#     (x, y, w, h) = joueurs[0] #Joueur 0 du haut
#     affichageJHaut=(max(0,x-decalageX), max(0,y-decalageY), w+2*decalageX, h+2*decalageY)
#     cv2.rectangle(frame1, (affichageJHaut[0], affichageJHaut[1]), (affichageJHaut[0]+affichageJHaut[2], affichageJHaut[1]+affichageJHaut[3]), (0, 200, 0), 2)
#     (x1, y1, w1, h1) = joueurs[1] #Joueur 1 du bas
#     affichageJBas=(x1-decalageX, y1-decalageY, w1+2*decalageX, h1+2*decalageY)
#     cv2.rectangle(frame1, (affichageJBas[0], affichageJBas[1]), (affichageJBas[0]+affichageJBas[2], affichageJBas[1]+affichageJBas[3]), (0, 255, 0), 2)

#     ###AFFICHAGE 
#     if(affichage):

#         cv2.imshow("feed", frame1)
#         if(devMode):cv2.imshow("feed2", dilated)

    ###CONTINUER LA LECTURE DE LA VIDEO
    # frame1 = frame2
    # frame2 = frame3
    # ret3, frame3 = cap.read()
    # #nbFrame+=1

    # if cv2.waitKey(40) == 27:
    #     break

cv2.destroyAllWindows()
cap.release()

print("fin")