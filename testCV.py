import cv2

image = cv2.imread("test2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

edged = cv2.Canny(blurred, 10, 100)

cv2.imshow("Original image", image)
cv2.imshow("Edged image", edged)
cv2.waitKey(0)
