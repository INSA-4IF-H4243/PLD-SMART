import copy
import cv2
#!pip install .
import cv2
import numpy as np
from smart.processor import ImageProcessor
from smart.video import Video, Image
from pynput import keyboard
import util
########################PARAMETRES :

devMode=False#mode Développeur (=voir les tous les contours, filtres...)
affichage=True#est-ce qu'on veut afficher les resultats ou juste enregistrer ?
enregistrementImage=True#Est-ce qu'on veut enregistrer la sortie en image ou juste en tableau de 0 et de 1
PixelSizeOutput=100#taille de la sortie (=entree du machine learning)
videoPath='dataset/clip/partie1.mp4'#chemin de la video
outPutPathJHaut='/jqi1'#chemin d'enregistrement de la silouhette du Joueur 1
outPutPathJBas='/jqi1'#chemin d'enregistrement de la silouhette du Joueur 2
outPutPath="img/"            #ex : avec les 3 outputs paths cela donnera : img/JHaut/nom_coup/outPutPathJHaut/liste des images du coup
fpsOutput=30#FPS de la sortie
cutFrameNB=30#nombre d'images pour un coups
videoResize=(800,400)#taille pour resize de la video pour traitement (petite taille = plus rapide) 

#taille de lentree du machine learning pour une seconde= fpsOutput * [PixelSizeOutput * PixelSizeOutput] (20*20*20=8000 pixels noir ou blanc)
#taille de lentree du machine learning = fpsOutput * [PixelSizeOutput * PixelSizeOutput] (20*20*20=8000 pixels noir ou blanc)
tableauSortieJHaut=[]
tableauSortieJBas=[]
tableau_position_balle = []
tableau_trajectoire_balle = []

tabCoups=["problemeDetection/PasClair...=>Poubelle","coup droit","revers","deplacement","service","immobile"]
k_pressed=False
def on_press(key):
        ###ENREGISTREMENT DONNEES pour les 7 dernières frames:
    if key==keyboard.Key.space:

        global k_pressed
        k_pressed=True

key_listener = keyboard.Listener(on_press=on_press)
key_listener.start()
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

#####LECTURE VIDEO
cap = cv2.VideoCapture(videoPath)
fps = cap.get(cv2.CAP_PROP_FPS)#FPS de la video d'entree
rapportFps=fps/fpsOutput
imageProcessor = ImageProcessor()

ret3, frame3 = cap.read()
#####AJUSTEMENT TAILLE
frame1=cv2.resize(frame3,videoResize)
milieu_y=150
milieu_x=400

#####INIT CONTOURS JOUEURS AU MILIEU DU TERRAIN (joeur 0 = joueur du haut, joueur 1 = joueur du bas)
joueurs=[(200,200,50,50),(200,200,50,50)]
print(joueurs)
balle = (milieu_x-25,milieu_y,50,50)
pos_balle = centre(balle)
pos_precedent = pos_balle
balle_detecte = False
rayon_detection = 10
compteur_non_detection = 0
limite = 3
#model_balle = ModelBalle.load_model_from_path('saved_models/model_balle_1.joblib')

#####LECTURE IMAGE PAR IMAGE
nbFrame=0
print("...")

factor = 0.49
parameters = {"substractor": {"history": 50, "threshold": 400},}

parameters_joueurs = {
    "filter": {"iterations": 10, "shape": (5, 5)},  # brush size
    "substractor": {"history": 200, "threshold": 200},
}

parameters_silouhette = {
    "filter": {"iterations": 3, "shape": (3, 3)},  # brush size
    "substractor": {"history": 200, "threshold": 200},
}

subtractors = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
subtractor = util.subtractor(subtractors[2], parameters["substractor"])
subtractor_joueurs = util.subtractor(subtractors[2], parameters_joueurs["substractor"])
ymin = 30
ymax = 370
xmin = 80
xmax = 720

while cap.isOpened() and ret3:#attention video qui s'arete au premier probleme dans la lecture a cause du resize
    ###AJUSTEMENT TAILLE
    # frame1=cv2.resize(frame1,videoResize)
    # frame2=cv2.resize(frame2,videoResize)

    transformations = []

    transformations.append(cv2.resize(frame3, (800,400)))
    # cv2.imshow("frame", transformations[-1])

    transformations.append(transformations[-1][ymin:ymax, xmin:xmax])
    # cv2.imshow("test", transformations[-1])

    transformations.append(cv2.cvtColor(transformations[-1], cv2.COLOR_BGR2GRAY))
    # cv2.imshow("gray", transformations[-1])

    transformations.append(subtractor.apply(transformations[-1]))
    transformations.append(util.filter(transformations[-1], "closing"))
    transformations.append(util.filter(transformations[-1], "dilation"))
    #cv2.imshow("gray", transformations[-1])


    

    

    # transformations.append(util.filter(transformations[-1], "dilation", parameters["filter"]))
    # cv2.imshow("dilation", transformations[-1])

    # transformations.append(cv2.medianBlur(transformations[-1], 5))
    # cv2.imshow("blur", transformations[-1])

    ###RECHERCHE CONTOURS DES FORMES EN MOUVEMENT
    contours, hierarchy = cv2.findContours(
    transformations[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tab_rec = []
    ball_rec =[]
    a_left=(ymax-ymin)/(150)
    b_left=ymax-a_left*xmax

    a_right=(ymin-ymax)/(150)
    b_right=ymax-a_right*xmin
    cv2.rectangle(transformations[0], (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.line(transformations[0],(xmax-150, int(a_left*(xmax-150)+b_left)), (xmax, int(a_left*(xmax)+b_left)), (255, 137, 0), 2)
    cv2.line(transformations[0],(xmin, int(a_right*(xmin)+b_right)),(xmin+150, int(a_right*(xmin+150)+b_right)), (137, 255, 0), 2)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1:
            x, y, w, h = cv2.boundingRect(contour)

            rec_c=( x, y, w, h )
            x_c, y_c = centre(rec_c)
            x_c=int(x_c)
            y_c=int(y_c)
            if area > 2000 : continue
            if area > 300 and (not  (h<20 or w<20)) and (y_c+ymin)>a_left*(x_c+xmin)+b_left and (y_c+ymin)>a_right*(x_c+xmin)+b_right:
                cv2.rectangle(transformations[0], (x_c+xmin, y_c+ymin), (x_c+xmin+5, y_c+ymin+5), (0, 0, 255), 2)
                tab_rec.append((x, y, w, h))
            elif area < 100:
                ball_rec.append((x, y, w, h))



    ###AFFICHAGE DE TOUS LES CONTOURS
    
    if devMode:
        for rec in tab_rec:
            (x, y, w, h) = rec
            cv2.rectangle(transformations[0], (x+xmin, y+ymin), (x+w+xmin, y+h+ymin), (0, 127, 127), 2)
    #print(len(tab_rec))
    ###CHOIX FINAL DES DEUX CONTOURS DES DEUX JOUEURS
    if(len(tab_rec)==2):       #Si à cette étape il n'y a que 2 contours, ce sont les bons
        if((tab_rec[0])[1]<(tab_rec[1])[1]):
            joueurs[0]=tab_rec[0]
            joueurs[1]=tab_rec[1]
        else:
            joueurs[0]=tab_rec[1]
            joueurs[1]=tab_rec[0]

    else:                       #Sinon on prend les contours les plus proches des anciens contours identifiés comme ceux des joueurs (=tracking)
        minJoueur0=(9999,9999,9999,9999)
        minJoueur1=(9999,9999,9999,9999)
        b0 = 0
        b1 = 0
        for rec in tab_rec: 
            #joueur du haut          
            if distance2(joueurs[0],rec) < distance2(joueurs[0],minJoueur0) and (centre(rec)[1]<milieu_y):
                #if distance2(joueurs[0],rec) < 5*5:
                    # if devMode:print("joueur0")
                    # if devMode:print(similarite(joueurs[0],rec))
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
            if distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 50 and distance2(balle,rec) > 1 :
                minBalle=rec  
                bBalle = 1
        if bBalle : balle = minBalle
        else : balle_detecte = False

    b = False
    if not balle_detecte:
        if compteur_non_detection < limite :
            for rec in ball_rec :
                if (distance2(balle,rec) < distance2(balle,minBalle) and distance2(balle,rec) < 50+rayon_detection*(compteur_non_detection+1)) :
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

        tab_prediction = []
        for point in tableau_position_balle :
            tab_prediction.append(point[0])
            tab_prediction.append(point[1])
        # print("ici")
        # print(len(tableau_position_balle))
        # resultat = model_balle.predict(tab_prediction)
        # print(resultat)
    
    for joueur in joueurs :
        cv2.rectangle(
                    transformations[0], (joueur[0] + xmin - 10, joueur[1] + ymin - 10), (joueur[0] + xmin + joueur[2], joueur[1] + ymin + joueur[3]), (0, 0, 255), 2
                )  # players (x,y)(x1,y1) -> (x,y,w,h)
    if balle_detecte : 
        cv2.rectangle(
                    transformations[0], (balle[0] + xmin - 10, balle[1] + ymin - 10), (balle[0] + xmin + balle[2], balle[1] + ymin + balle[3]), (0, 255, 0), 2
                )  # ball
    
    x_bas=round(joueurs[1][0] + xmin - 10)
    y_bas=round(joueurs[1][1] + ymin - 10)
    w_bas=round(joueurs[1][0] + xmin + joueurs[1][2] - x_bas)
    h_bas=round(joueurs[1][1] + ymin + joueurs[1][3] - y_bas)
    affichageJBas=(x_bas, y_bas, w_bas, h_bas)

    x_haut=round(joueurs[0][0] + xmin - 10)
    y_haut=round(joueurs[0][1] + ymin - 10)
    w_haut=round(joueurs[0][0] + xmin + joueurs[0][2] - x_haut)
    h_haut=round(joueurs[0][1] + ymin + joueurs[0][3] - y_haut)
    affichageJHaut=(x_haut, y_haut, w_haut, h_haut)
    ###DESSIN DU CONTOUR DES JOUEURS)
    cv2.rectangle(
        transformations[0], (x_bas,y_bas),(x_bas + w_bas, y_bas + h_bas), (255, 0, 255), 2)
    cv2.rectangle(
        transformations[0], (x_haut,y_haut),(x_haut + w_haut, y_haut + h_haut) , (0, 255, 255), 2)        

    ###RECUPERATION SILOUHETTE 
    if(nbFrame%rapportFps<1):
        (x, y, w, h) = affichageJHaut
        (x1, y1, w1, h1) = affichageJBas

        transfomations_joueurs = []
        transfomations_joueurs.append(cv2.resize(frame3, (800,400)))
        transfomations_joueurs.append(transfomations_joueurs[-1][ymin:ymax, xmin:xmax])
        transfomations_joueurs.append(cv2.cvtColor(transfomations_joueurs[-1], cv2.COLOR_BGR2GRAY))
        transfomations_joueurs.append(subtractor_joueurs.apply(transfomations_joueurs[-1]))
        # cv2.rectangle(
        #     transfomations_joueurs[3], (joueurs[0][0],joueurs[0][1]),(joueurs[0][0] + joueurs[0][2], joueurs[0][1] + joueurs[0][3]), (255, 255, 255), 2)
        # cv2.rectangle(
        #     transfomations_joueurs[3], (joueurs[1][0],joueurs[1][1]),(joueurs[1][0] + joueurs[1][2], joueurs[1][1] + joueurs[1][3]) , (255, 255, 255), 2) 
        try:

            # crop_imgBas = imageProcessor.crop_frame_shadow_player(transformations[0], x1, x1+w1, y1, y1+h1)
            # crop_imgHaut = imageProcessor.crop_frame_shadow_player(transformations[0], x, x+w, y, y+h)
            # silouhetteHaut = imageProcessor.resize_img(crop_imgHaut,(50, 50), interpolation=cv2.INTER_LINEAR)  
            # silouhetteBas = imageProcessor.resize_img(crop_imgBas, (50, 50), interpolation=cv2.INTER_LINEAR)
            
            silouhetteBas  = transfomations_joueurs[3][max(0,joueurs[1][1]-15):joueurs[1][1] + joueurs[1][3]+30,max(0,joueurs[1][0]-15):joueurs[1][0] + joueurs[1][2]+30]
            
            silouhetteBas=np.ceil(silouhetteBas/255)*255
            silouhetteBas=util.filter(silouhetteBas, "closing",parameters_silouhette["filter"])
            silouhetteBas=cv2.resize(silouhetteBas,(PixelSizeOutput,PixelSizeOutput))
        except:
            silouhetteBas = np.zeros((PixelSizeOutput,PixelSizeOutput))

        try:
            silouhetteHaut = transfomations_joueurs[3][max(joueurs[0][1]-15,0):joueurs[0][1] + joueurs[0][3]+30,max(0,joueurs[0][0]-15):joueurs[0][0] + joueurs[0][2]+30]
            silouhetteHaut=np.ceil(silouhetteHaut/255)*255
            silouhetteHaut=util.filter(silouhetteHaut, "closing",parameters_silouhette["filter"])
            silouhetteHaut=cv2.resize(silouhetteHaut,(PixelSizeOutput,PixelSizeOutput))
        except:
            silouhetteHaut = np.zeros((PixelSizeOutput,PixelSizeOutput))


    ###AFFICHAGE 
    
    
        cv2.imshow("feed", transformations[0])
        #if(devMode):cv2.imshow("feed2", dilated)
        if devMode:cv2.imshow("closing", transformations[3])
        cv2.imshow("JoueurHaut : ", silouhetteHaut)
        cv2.imshow("JoueurBas : ", silouhetteBas)

    

        ##ENREGISTREMENT
        if(k_pressed==True):
            print("dernier coup du joueur en haut:")
            for i in range(len(tabCoups)):
                print(i," : ",tabCoups[i])
            coupJHaut=int(input())

            print("dernier coup du joueur en bas:")
            for i in range(len(tabCoups)):
                print(i," : ",tabCoups[i])
            coupJBas=int(input())

            if(coupJHaut):imageProcessor.save_ImageList(tableauSortieJHaut[len(tableauSortieJHaut)-cutFrameNB:len(tableauSortieJHaut)],outPutPath+"JHaut/"+tabCoups[coupJHaut]+outPutPathJHaut+str(nbFrame),enregistrementImage)
            if(coupJBas):imageProcessor.save_ImageList(tableauSortieJBas[len(tableauSortieJHaut)-cutFrameNB:len(tableauSortieJHaut)],outPutPath+"JBas/"+tabCoups[coupJBas]+outPutPathJHaut+str(nbFrame),enregistrementImage)
            print("\nséquence enregistrée, reprise...\n")
            k_pressed=False

        ###ENREGISTREMENT des silouhettes dans le TABLEAU
        tableauSortieJHaut.append(silouhetteHaut/255)
        tableauSortieJBas.append(silouhetteBas/255)

    ###CONTINUER LA LECTURE DE LA VIDEO
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