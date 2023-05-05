from skimage.restoration import estimate_sigma
import cv2
import numpy as np
from ..video.Video import Video
import os


def estimate_noise(img):
    """
    Parameters
    ----------
    img : ndarray
        Input image 3-dim

    Returns
    -------
    float
        Estimate of the noise standard deviation of the image
    """
    return estimate_sigma(img, average_sigmas=True, channel_axis=-1)


class ImageProcessor:
    def __init__(self):
        pass

    def remove_background(self, input_img):
        """
        Parameters
        ----------
        input_img : np.ndarray 3-dim
            Input image

        Returns
        -------
        np.ndarray 3-dim
            Image with background removed
        """
        # Blur to image to reduce noise
        input_img = cv2.GaussianBlur(input_img, (5, 5), 0)

        # We bin the pixels. Result will be a value 1..5
        bins = np.array([0, 51, 102, 153, 204, 255])
        input_img[:, :, :] = np.digitize(input_img[:, :, :], bins, right=True) * 51

        # Create single channel greyscale for thresholding
        input_img_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # Perform Otsu thresholding and extract the background.
        # We use Binary Threshold as we want to create an all white background
        ret, background = cv2.threshold(input_img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert black and white back into 3 channel greyscale
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Perform Otsu thresholding and extract the foreground.
        # We use TOZERO_INV as we want to keep some details of the foregorund
        ret, foreground = cv2.threshold(input_img_grey, 0, 255,
                                        cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)  # Currently foreground is only a mask
        foreground = cv2.bitwise_and(input_img, input_img,
                                     mask=foreground)  # Update foreground with bitwise_and to extract real foreground

        # Combine the background and foreground to obtain our final image
        finalimage = background + foreground

        return finalimage

    def crop_image(self, img, start_x: int, end_x: int, start_y: int, end_y: int):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Input image
        start_x : int
            Starting x coordinate
        end_x : int
            Ending x coordinate
        start_y : int
            Starting y coordinate
        end_y : int
            Ending y coordinate

        Returns
        -------
        Image : np.ndarray 3-dim
            Cropped image
        """
        return img[start_y:end_y, start_x:end_x]

    def save_img(self, img, path: str):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Image to save
        path : str
            Path to save the image
        """
        cv2.imwrite(path, img)
        return

    def crop_shadow_player_save(self, video: Video, nb_start: int, nb_end: int,
                                start_x: int, end_x: int,
                                start_y: int, end_y: int,
                                folder_path: str):
        """
        Crop the shadow player from the video and save the cropped images to a folder

        Parameters
        ----------
        video: Video
            The video to crop
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
        list_frames = video.frames[nb_start:nb_end]
        for i in range(nb_start, nb_end):
            crop_img = self.crop_image(list_frames[i], start_x, end_x, start_y, end_y)
            no_bg_img = self.remove_background(crop_img)
            gray_img = cv2.cvtColor(no_bg_img, cv2.COLOR_BGR2GRAY)
            inverted_img = cv2.bitwise_not(gray_img)
            _, thresh = cv2.threshold(inverted_img, 0, 255, cv2.THRESH_BINARY)
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, thresh)
        return
