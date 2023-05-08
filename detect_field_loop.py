import cv2
import numpy as np
from court_reference import CourtReference

def find_rows(points):
    rows = list()
    for pt in points:
        is_found = False
        for row in rows:
            if abs(row[1] - pt[1]) < 20:
                row[0].append(pt)
                is_found = True
        if not is_found:
            rows.append(([pt], pt[1])) 
    return rows

def find_extrema(points):
    extrema = list()
    rows = find_rows(points)
    y_max_idx, y_min_idx = 0, 0
    y_max, y_min = 0, 10000
    for idx, row in enumerate(rows):
        if row[1] > y_max:
            y_max = row[1]
            y_max_idx = idx
        if row[1] < y_min:
            y_min = row[1]
            y_min_idx = idx
    row_top = rows[y_min_idx][0]
    row_top = sorted(row_top, key=lambda x: x[0])
    row_bottom = rows[y_max_idx][0]
    row_bottom = sorted(row_bottom, key=lambda x: x[0])
    extrema = [row_top[0], row_top[-1], row_bottom[0], row_bottom[-1]]
    return extrema

def draw_lines(img, lines, trasnfo):
    for line in lines:
        x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        cv2.line (img, (x1+transfo[0], y1+transfo[1]), (x2+transfo[0], y2+transfo[1]), (0, 255, 0), 2)
    return img

def perspective_transform(pt: list, matrix: 'np.array'):
    pt.append(1) # [x, y, 1]
    pt_arr = np.asarray(pt) # np.array([x, y, 1])
    pt_transformed = np.dot(matrix, pt_arr) # p' = H.p
    pt_transformed = pt_transformed / pt_transformed[2] # p' = np.array([x', y', 1])
    return (round(pt_transformed[0]), round(pt_transformed[1]))

def transform_lines(lines: list, matrix: 'np.array'):
    transformed_lines = list()
    for line in lines:
        pt_a = list(line[0])
        pt_b = list(line[1])
        pt_a = perspective_transform(pt_a, matrix)
        pt_b = perspective_transform(pt_b, matrix)
        transformed_lines.append((pt_a, pt_b))
    return transformed_lines

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

input_video = cv2.VideoCapture('input/clip_problems2.mp4')

width = 1200
height = 600

fps = int(input_video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output/clip_problems2.mp4', fourcc, fps, (width, height))

court_ref = CourtReference()

ret, frame = input_video.read()
comp = cv2.resize(frame, (width, height))
gray_comp = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)

while input_video.isOpened():
    if ret == True:
        frame = cv2.resize(frame, (width, height))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        error = mse(gray_frame, gray_comp)

        if(error <= 15):
            transfo = [round(0.15*width), round(0.2*height), round(0.6*height), round(0.7*width)] #x, y, h, w
            court = frame[transfo[1]:transfo[1]+transfo[2],transfo[0]:transfo[0]+transfo[3]]

            gray = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
            lines = cv2.HoughLinesP(image=gray, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=round(0.25*width), maxLineGap=20)

            if((lines is not None)):
                if(len(lines)>8):
                    lines = np.squeeze(lines)
                    points = list()
                    for line in lines: 
                        points.extend([line[:2], line[2:]])
                    extrema = find_extrema(points)
                    reference = [court_ref.baseline_top[0], court_ref.baseline_top[1], court_ref.baseline_bottom[0], court_ref.baseline_bottom[1]]
                    matrix, _ = cv2.findHomography(np.float32(reference), np.float32(extrema), method=0)
                    transformed_lines = transform_lines(court_ref.court_lines, matrix)
                    frame = draw_lines(frame, transformed_lines, transfo)   
                    cv2.imshow("court", frame) 
                    output_video.write(frame)

        ret, frame = input_video.read()

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

  
test1 = draw_lines(frame, transformed_lines, transfo) 
cv2.destroyAllWindows()
input_video.release()