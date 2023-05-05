from rembg import remove, new_session
from skimage.restoration import estimate_sigma
import cv2
import PIL
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

    def remove_background(self, input_img, threshold = 0.3, mode_img = "RGB"):
        """
        Parameters
        ----------
        input_img : np.ndarray 3-dim
            Input image
        threshold : float, optional
            Threshold to determine which model to use, by default 0.3

        Returns
        -------
        np.ndarray 3-dim
            Image with background removed
        """
        model_name = "u2net_human_seg" if (estimate_noise(input_img) < threshold) else "u2netp"
        session = new_session(model_name=model_name)
        output = remove(input_img, session=session, post_process_mask=True)
        img_output = PIL.Image.fromarray(output)
        img_output = img_output.convert(mode_img)
        return np.array(img_output)
    
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
    
    def crop_silouhette(self, img):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Input image
        """
        miny=len(img)
        maxy=0
        minx=len(img[0])
        maxx=0
        
        vide=True
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j]==255):
                    vide=False
                    if(i>maxy):maxy=i
                    if(i<miny):miny=i

                    if(j>maxx):maxx=j
                    if(j<minx):minx=j
        if(not vide):
            img2=self.crop_image(img,minx, maxx,miny,maxy)
            img3=cv2.resize(img2,(20,20))
            return img3
        else:
            return False
    
    
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
                                folder_path: str, threshold: float = 1.1,
                                mode_img: str = "RGB"):
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
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            no_bg_img = self.remove_background(gray_img, threshold, mode_img)
            _, thresh = cv2.threshold(no_bg_img, 0, 255, cv2.THRESH_BINARY)
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, thresh)
        return
    
'''
def remove_background_old(image):
    # Thresholding the image
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    thresh = cv.inRange(image, lower, upper)

    # Apply morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # Invert the image
    mask = 255 - morph
    result = cv.bitwise_and(image, image, mask=mask)

    # Save the results
    cv.imwrite("..\\TestRemoveBackground\\thresh.png", thresh)
    cv.imwrite("..\\TestRemoveBackground\\morph.png", morph)
    cv.imwrite("..\\TestRemoveBackground\\mask.png", mask)
    cv.imwrite("..\\TestRemoveBackground\\result.png", result)

    # Show the results
    cv.imshow('thresh', thresh)
    cv.imshow('morph', morph)
    cv.imshow('mask', mask)
    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return result


# Remove background adaptively
def remove_background_adaptive_old(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 25)
    cv.imwrite("..\\TestRemoveBackground\\thresh1.png", thresh1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    morph = cv.morphologyEx(thresh1, cv.MORPH_DILATE, kernel)
    division = cv.divide(gray, morph, scale=255)
    thresh2 = cv.threshold(division, 0, 255, cv.THRESH_OTSU)[1]
    cv.imwrite("..\\TestRemoveBackground\\thresh2.png", thresh2)
    return thresh2
'''
