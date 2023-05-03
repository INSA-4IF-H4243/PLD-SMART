import cv2
import numpy as np

img = cv2.imread('frames/frame21.jpg')
cv2.imshow('Original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 30, 200)
print(canny)
contours, hierarchy = cv2.findContours(canny,
   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours = " ,len(contours))
cv2.imshow('Canny Edges', canny)

# closed_contours = []

# for cnt in contours:
#    if cv2.isContourConvex(cnt) == True:
#       closed_contours.append(cnt)
#    else:
#       pass
# cv2.drawContours(img, closed_contours, -1, (0, 255, 0), 3)
import random
i =0
for contour in contours:
   i+=1
   test=cv2.arcLength(contour,True)
   #print(test)
   if(test>10):
      color = tuple(np.random.random(size=3) * 256)
      cv2.drawContours(img, contour, -1, color, 3)
   
cv2.imshow('Contours', img)
print(i)
cv2.waitKey(0)
cv2.destroyAllWindows()