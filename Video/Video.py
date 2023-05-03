import cv2
import numpy as np


class Video(object):
    def __init__(self, path: str, frames):
        self.path = path
        self.frames = np.array(frames)

    @classmethod
    def read_video(video, path: str):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            frames.append(frame)
        cap.release()
        return video(path, frames)

    def save_video(self, frames):
        cv2.imwrite(self.path, frames)
        return
