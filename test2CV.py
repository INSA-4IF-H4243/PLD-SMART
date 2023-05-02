import cv2
import numpy as np

img = cv2.imread('test2.png')
cv2.imshow('Original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 30, 200)

contours, hierarchy = cv2.findContours(canny,
   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours = " ,len(contours))
cv2.imshow('Canny Edges', canny)


cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()