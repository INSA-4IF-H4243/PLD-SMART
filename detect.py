import cv2
import numpy as np
import argparse

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

def find_contour_joueur(frame1, frame2, up_down, middle):
    diff = cv2.absdiff(frame1, frame2)
    if(up_down=='up'): diff = diff*2
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13,13), 0)
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

# def detect_field(court): 
#     gray= cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 100, 600, apertureSize = 3)
#     lines = cv2.HoughLinesP(edges, 1, (np.pi/2)*0.1, 20, None, 0, 50)

#     for line in lines:
#         pt1 = (line[0][0], line[0][1])
#         pt2 = (line[0][2], line[0][3])
#         cv2.line(court, pt1, pt2, (0,255,0), 2)

#     return court

parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path

input_video = cv2.VideoCapture(input_video_path)

fps = int(input_video.get(cv2.CAP_PROP_FPS))
output_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))


ret, frame1 = input_video.read()
ret, frame2 = input_video.read()

while input_video.isOpened():
    if ret == True:
        y=len(frame1)
        frame1_up=frame1[0:round(y/2)]
        frame1_down=frame1[round(y/2):round(y)]
        frame2_up=frame2[0:round(y/2)]
        frame2_down=frame2[round(y/2):round(y)]
        middle =  round(y/2)
        frame_up=find_contour_joueur(frame1_up, frame2_up, 'up', middle)
        frame_down=find_contour_joueur(frame1_down, frame2_down, 'down', middle)

        cond1 = frame_up is None
        cond2 = frame_down is None
        if(not cond1 and not cond2):
            frame=np.concatenate((frame_up, frame_down))
        # frame = detect_field(frame)

        cv2.imshow("window", frame)

        output_video.write(frame)

        frame1 = frame2
        ret, frame2 = input_video.read()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
input_video.release()
