import cv2
import numpy as np
import os

class Video(object):
    def __init__(self, **kwargs):
        try:
            self.path = kwargs['path']
            cap = cv2.VideoCapture(self.path)
            frames = []
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                frames.append(frame)
            cap.release()
            self.frames = np.array(frames)
        except:
            self.path = ''
            self.frames = np.array([])

        try:
            self.frames = kwargs['frames']
            self.frames = np.array(self.frames)
        except KeyError:
            raise ValueError("The frames are not defined")
        else:
            if len(self.frames) == 0:
                raise ValueError("The frames are empty")

    @classmethod
    def read_video(video, path: str = ''):
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist".format(path))
        kwargs = {'path': path}
        return video(**kwargs)
    
    @classmethod
    def read_video_from_frames(video, frames):
        kwargs = {'frames': frames}
        return video(**kwargs)

    def save_video(self, saved_path: str):
        ## Write the `RGB` frames to a video, using `cv2.VideoWriter`
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = self.frames[0].shape[:2]
        out = cv2.VideoWriter(saved_path, fourcc, 20.0, size)

        num_frames = len(self.frames)
        for i in range(num_frames):
            img = cv2.cvtColor(self.frames[i], cv2.COLOR_RGB2BGR)
            out.write(img)
        out.release()
        return
    
    def save_all_frames(self, folder_path: str =''):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in range(len(self.frames)):
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, self.frames[i])
        return
    
    def save_some_frames(self, list_nb_frames, folder_path: str =''):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in list_nb_frames:
            if i < 10:
                saved_path = os.path.join(folder_path, 'frame_00{}.jpg'.format(i))
            elif i >= 10 and i < 100:
                saved_path = os.path.join(folder_path, 'frame_0{}.jpg'.format(i))
            else:
                saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, self.frames[i])
        return
        