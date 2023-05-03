import cv2
import numpy as np

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

def find_contour_joueur(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tab_rec = []
    for contour in contours:
        rec_base = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 150:
            continue
        
        for rec in tab_rec:
            if superposition(rec_base, rec):
                rec_base = englobant(rec_base, rec)
                tab_rec.remove(rec)
        tab_rec.append(rec_base)

    for rec in tab_rec:
        (x, y, w, h) = rec
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    
    return(frame1)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('v.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

print(frame1)

y=len(frame1)
frame1_up=frame1[0:y/2]
frame1_down=frame1[y/2:y]
frame2_up=frame2[0:y/2]
frame2_down=frame2[y/2:y]

cv2.imshow("2", frame1_up)

while cap.isOpened():

    y=len(frame1)
    frame1_up=frame1[0:y/2]
    frame1_down=frame1[y/2:y]
    frame2_up=frame2[0:y/2]
    frame2_down=frame2[y/2:y]

    frame_up=find_contour_joueur(frame1_up,frame2_up)
    frame_down=find_contour_joueur(frame1_down,frame2_down)

    frame=np.concatenate((frame_up, frame_down))
    cv2.imshow("feed", frame)

    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()