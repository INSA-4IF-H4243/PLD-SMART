import cv2
import numpy as np

class Image(object):
    def __init__(self, path, image) -> None:
        self.path = path
        self.image = np.array(image)

    @classmethod
    def load_image(obj, path: str):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError("Image not found!")
        return obj(path, image)
    
    def save_image(self, path: str):
        cv2.imwrite(path, self.image)
        return