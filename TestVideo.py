import cv2
vidcap = cv2.VideoCapture('v.mp4')
success,image = vidcap.read()
count = 0
b = 0
images=[]
while (True):
    if(success):
        images.append(image)
        b=0
    elif (b <= 3):
        b+=1
        print('frame is false')
    else:
        break
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100)) # added this line
    success,image = vidcap.read()
    
    print('Read a new frame: ', success)
    count += 1
print(len(images))
print(images[30][0][0])
cv2.imshow(images[30])
