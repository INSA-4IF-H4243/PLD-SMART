import cv2
import numpy as np

# court = cv2.imread('input/tennis_court.webp')
input_video = cv2.VideoCapture('input/clip_usopen.mp4')
ret, court = input_video.read()
court = cv2.resize(court, (800, 500)) 
y1 = 125
x1 = 150
h = 285
w = 500
court2 = court[y1:y1+h,x1:x1+w]
cv2.imshow('court2',court2)
# Select ROI
# r = cv2.selectROI("select the area", court)
  
# Crop image
# court = court[int(r[1]):int(r[1]+r[3]), 
#                       int(r[0]):int(r[0]+r[2])]
  

# input_video = cv2.VideoCapture('input/input5.mp4')
# ret, court = input_video.read()
# y = len(court)
# court = court[250:round(y-200)]


gray= cv2.cvtColor(court2, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(court2, 100, 600, apertureSize = 3)
# cv2.imshow('court',edges)

# lines = cv2.HoughLinesP(edges, 1, (np.pi/2)*0.1, 20, None, 0, 50)
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=100,maxLineGap=100)
# lines = cv2.detect(edges, 1, (np.pi/2)*0.1, 20, None, 0, 50)

for line in lines:
    pt1 = (line[0][0], line[0][1])
    pt2 = (line[0][2], line[0][3])
    cv2.line(court, (pt1[0]+x1, pt1[1]+y1), (pt2[0]+x1, pt2[1]+y1), (0,255,0), 2)

# cv2.imshow('court',court)
cv2.waitKey()

        
