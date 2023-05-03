import cv2
import numpy as np


def superposition(rec1, rec2):  # returns None if rectangles don't intersect
#(x, y, w, h)
    dx = min(rec1[0]+rec1[2], rec2[0]+rec2[2]) - max(rec1[0], rec2[0])
    dy = min(rec1[1]+rec1[3], rec2[1]+rec2[3]) - max(rec1[1], rec2[1])
    if (dx>=0) and (dy>=0):
        return True
    
def englobant(rec1,rec2):
    x1=min(rec1[0],rec2[0])
    x2=max(rec1[0]+rec1[2],rec2[0]+rec2[2])
    y1=min(rec1[1],rec2[1])
    y2=max(rec1[1]+rec1[3],rec2[1]+rec2[3])   
    w=x2-x1
    h=y2-y1
    rec3=(x1,y1,w,h)
    return rec3

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('v.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    tabRec=[]
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        recNew=(x, y, w, h)
        for rec in tabRec:
            if superposition(rec,recNew):
                recNew=englobant(rec,recNew)

        if cv2.contourArea(contour) < 100:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break


cap.release()