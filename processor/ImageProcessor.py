import cv2
import numpy as np
from video.Video import Video
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

class ImageProcessor:
    def __init__(self):
        pass

    def remove_background(self, input_img, threshold=0.8):
        """
        Parameters
        ----------
        input_img : np.ndarray 3-dim
            Input image
        threshold : float, optional
            Threshold to remove the background, by default 0.8
        Returns
        -------
        np.ndarray 3-dim
            Image with background removed
        """
        segmentor = SelfiSegmentation()
        final_image = segmentor.removeBG(input_img, (0, 0, 0), threshold=threshold)
        return final_image

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

    def resize_img(self, img, new_shape: tuple, interpolation=cv2.INTER_BITS):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Input image
        new_shape : tuple
            New shape of the image

        Returns
        -------
        Image : np.ndarray 3-dim
            Resized image
        """
        return cv2.resize(img, new_shape, interpolation=interpolation)

    def flouter_image(self, image):
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 4)
        ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def binary(self, image):
        """
        Parameters
        ----------
        img : greyImage 2 dim
            Input image
        Returns
        ------
        Image : black, and white if pixel !=0
        """
        _, new_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        return new_img

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

    def crop_all_shadows_player_save(self, video: Video, nb_start: int, nb_end: int,
                                     start_x: int, end_x: int,
                                     start_y: int, end_y: int,
                                     folder_path: str, threshold=0.8):
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
            crop_img = self.crop_image(list_frames[i],
                                       start_x, end_x, start_y, end_y)
            no_bg_img = self.remove_background(crop_img, threshold=threshold)
            gray_img = cv2.cvtColor(no_bg_img, cv2.COLOR_BGR2GRAY)
            _, final_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
            saved_path = os.path.join(folder_path, 'frame_{}.jpg'.format(i))
            cv2.imwrite(saved_path, final_img)
        return

    def crop_frame_shadow_player(self, frame, start_x, end_x, start_y, end_y, threshold=0.8):
        """
        Crop the shadow player from the video and save the cropped images to a folder

        Parameters
        ----------
        frame: np.ndarray 3-dim
            The frame to crop
        start_x: int
            The starting x coordinate of the cropped image
        end_x: int
            The ending x coordinate of the cropped image
        start_y: int
            The starting y coordinate of the cropped image
        end_y: int
            The ending y coordinate of the cropped image
        threshold: float
            The threshold to remove the background

        Returns
        -------
        final_img: np.ndarray 2-dim
            Cropped shadow image
        """
        crop_img = self.crop_image(frame, start_x, end_x, start_y, end_y)
        no_bg_img = self.remove_background(crop_img, threshold=threshold)
        gray_img = cv2.cvtColor(no_bg_img, cv2.COLOR_BGR2GRAY)
        _, final_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
        return final_img

    def save_ImageList(self, imageList, outPutPath, toImageBool):
        """
        save a array of black and white image to 1 and 0 in csv

        Parameters
        ----------
        imageList: List of matrix 2d
            the list of image
        outPutPath: string
           folder
        toImageBool: boolean
            1 if you want to save the images in a /image folder, 0 if you just want the 1/0 csv
        """
        count = 0
        outPutImBas = outPutPath+'/images'
        if not os.path.exists(outPutPath):
            os.makedirs(outPutPath)
        if toImageBool and not os.path.exists(outPutImBas):
            os.makedirs(outPutImBas)

        for i in imageList:
            count += 1
            saved_path = os.path.join(outPutPath, 'frame_{}.csv'.format(count))
            i = np.asmatrix(i)
            i = i.astype(int)
            np.savetxt(saved_path, i, fmt='%d', delimiter=" ")
            if(imageList):
                if count < 10:
                    saved_pathIm = os.path.join(
                        outPutImBas, 'frame_0{}.jpg'.format(count))
                    cv2.imwrite(saved_pathIm, i*255)
                else:
                    saved_pathIm = os.path.join(
                        outPutImBas, 'frame_{}.jpg'.format(count))
                    cv2.imwrite(saved_pathIm, i*255)

    def crop_silouhette(self, img, pixelSize):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Input image
        Returns
        ------
        Image : np.ndarray 3-dim
            silouhette image (grey)
        """
        miny = len(img)
        maxy = 0
        minx = len(img[0])
        maxx = 0

        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] != 0):
                    if(i > maxy):
                        maxy = i
                    if(i < miny):
                        miny = i

                    if(j > maxx):
                        maxx = j
                    if(j < minx):
                        minx = j

        if(miny < maxy and maxx > minx):
            img2 = self.crop_image(img, minx, maxx, miny, maxy)
            img3 = cv2.resize(img2, (pixelSize, pixelSize))
            img = img3
        return img
