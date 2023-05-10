import cv2
import numpy as np
import os

class Image:
    def __init__(self, flags, **kwargs):
        try:
            self.path = kwargs['path']
            self.img = cv2.imread(self.path, flags=flags)
            self.img = np.array(self.img)
            self.width = self.img.shape[1]
            self.height = self.img.shape[0]
        except:
            self.path = ''
            self.img = np.array([])
            self.width = 0
            self.height = 0

    @classmethod
    def load_image(image, flags, path: str = ''):        
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist".format(path))
        kwargs = {'path': path}
        return image(flags, **kwargs)
    
    def save_image(self, saved_path: str):
        cv2.imwrite(saved_path, self.img)
        return
