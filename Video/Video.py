import cv2
import numpy as np
import os
from ..processor.ImageProcessor import ImageProcessor

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

    @classmethod
    def read_video(video, path: str = ''):
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist".format(path))
        kwargs = {'path': path}
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
    
    def save_some_frames(self, nb_start: int = 0, nb_end: int = 30, folder_path: str =''):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in range(nb_start, nb_end):
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, self.frames[i])
        return
    
    def crop_shadow_player_save(self, nb_start: int, nb_end: int,
                                start_x: int, end_x: int,
                                start_y: int, end_y: int,
                                folder_path: str, threshold: float = 1.1):
        """
        Crop the shadow player from the video and save the cropped images to a folder
        
        Parameters
        ----------
        nb_start: int
            The number of the first frame to be cropped
        nb_end: int
            The number of the last frame to be cropped
        start_x: int
            The starting x coordinate of the cropped image
        end_x: int
            The ending x coordinate of the cropped image
        start_y: int
            The starting y coordinate of the cropped image
        end_y: int
            The ending y coordinate of the cropped image
        folder_path: str
            The path to the folder where the shadow images will be saved
        threshold: float
            The threshold to remove the background
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_processor = ImageProcessor()
        for i in range(nb_start, nb_end):
            crop_img = image_processor.crop_image(self.frames[i], start_x, end_x, start_y, end_y)
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            no_bg_img = image_processor.remove_background(gray_img, threshold)
            _, thresh = cv2.threshold(no_bg_img, 0, 255, cv2.THRESH_BINARY)
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, thresh)
        return
        