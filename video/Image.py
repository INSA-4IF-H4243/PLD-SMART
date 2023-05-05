import PIL
import cv2
import numpy as np
import os

class Image:
    def __init__(self, path: str = ''):
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist".format(path))
        self.path = path
        self.img = PIL.Image.open(path) # self.img array 3-dim
        self.img = np.array(self.img)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.img_imread = cv2.imread(path)

    @classmethod
    def load_image(image, path: str = ''):
        if not os.path.exists(path):
            raise FileNotFoundError("The file {} does not exist".format(path))
        kwargs = {'path': path}
        return image(**kwargs)

    def save_image(self, saved_path: str):
        cv2.imwrite(saved_path, self.img)
        return

