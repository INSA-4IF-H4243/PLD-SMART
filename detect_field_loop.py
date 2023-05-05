import cv2
import numpy as np

input_video = cv2.VideoCapture('input/clip_problems.mp4')

width = 1600
height = 800

fps = int(input_video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output/clip_problems.mp4', fourcc, fps, (width, height))

ret, frame = input_video.read()

while input_video.isOpened():
    if ret == True:
        frame = cv2.resize(frame, (width, height)) 
        x = round(0.15*width)
        y = round(0.2*height)
        h = round(0.6*height)
        w = round(0.7*width)
        court = frame[y:y+h,x:x+w]

        gray= cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(court, 100, 700, apertureSize = 3)

        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=100,maxLineGap=100)

        if(lines is not None):
            if(len(lines)>10):
                for line in lines:
                    pt1 = (line[0][0], line[0][1])
                    pt2 = (line[0][2], line[0][3])
                    cv2.line(frame, (pt1[0]+x, pt1[1]+y), (pt2[0]+x, pt2[1]+y), (0,255,0), 2)
                cv2.imshow("court", frame) 
                output_video.write(frame)

        ret, frame = input_video.read()

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
input_video.release()

        
