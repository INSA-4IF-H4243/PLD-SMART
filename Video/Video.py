import cv2

class Video(object):
    def __init__(self, path):
        self.path = path

    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)
        cap.release()
        return frames

    def save_video(self, frames):
        cv2.imwrite(self.path, frames)
        return