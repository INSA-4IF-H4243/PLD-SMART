from rembg import remove, new_session
from skimage.restoration import estimate_sigma
import cv2
import numpy as np


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

    # Remove background using rembg
    def remove_background(self, input_img, threshold=0.3):
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
        # model_name = "u2net_human_seg" if (estimate_noise(input_img) < threshold) else "u2netp"
        model_name = "isnet_general_use"
        session = new_session(model_name=model_name)
        output = remove(input_img, session=session,
                        post_process_mask=True, alpha_matting=True,
                        alpha_matting_foreground_threshold=270,
                        alpha_matting_background_threshold=20,
                        alpha_matting_erode_structure_size=11)
        return output

    '''
    --------------------------------------------------------------------------------------------------------------------
    STRATEGIES FOR REMOVING BACKGROUND USING OPENCV INSTEAD OF REMBG
    (which uses Machine Learning models and takes too much time)
    Credits: https://www.freedomvc.com/index.php/2022/01/17/basic-background-remover-with-opencv/
    --------------------------------------------------------------------------------------------------------------------
    '''

    '''
    Strategy 1 (Longest runtime)
    
    1. Perform Gaussian Blur to remove noise
    2. Simplify our image by binning the pixels into six equally spaced bins in RGB space. In other words convert into a
       5 x 5 x 5 = 125 colors
    3. Convert our image into greyscale and apply Otsu thresholding to obtain a mask of the foreground
    4. Apply the mask onto our binned image keeping only the foreground (essentially removing the background)
    '''

    def remove_background_1(self, input_img):
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

    '''
    Strategy 2 (10x faster than Strategy 1, lowest quality): OpenCV2 Simple Thresholding
    
    1. Convert our image into Greyscale
    2. Perform simple thresholding to build a mask for the foreground and background
    3. Determine the foreground and background based on the mask
    4. Reconstruct original image by combining foreground and background
    '''

    def remove_background_2(self, input_img):
        # First Convert to Grayscale
        myimage_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        ret, baseline = cv2.threshold(myimage_grey, 127, 255, cv2.THRESH_TRUNC)

        ret, background = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY)

        ret, foreground = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY_INV)

        foreground = cv2.bitwise_and(input_img, input_img,
                                     mask=foreground)  # Update foreground with bitwise_and to extract real foreground

        # Convert black and white back into 3 channel greyscale
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Combine the background and foreground to obtain our final image
        finalimage = background + foreground
        return finalimage

    ''' 
    Strategy 3 (10x faster than Strategy 1, lower quality): Working in HSV Color Space
    
    1. Convert our image into HSV color space
    2. Perform simple thresholding to create a map using Numpy based on Saturation and Value
    3. Combine the map from S and V into a final mask
    4. Determine the foreground and background based on the combined mask
    5. Reconstruct original image by combining extracted foreground and background
    '''

    def remove_background_3(self, input_img):
        # BG Remover 3
        myimage_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

        # Take S and remove any value that is less than half
        s = myimage_hsv[:, :, 1]
        s = np.where(s < 127, 0, 1)  # Any value below 127 will be excluded

        # We increase the brightness of the image and then mod by 255
        v = (myimage_hsv[:, :, 2] + 127) % 255
        v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask

        # Combine our two masks based on S and V into a single "Foreground"
        foreground = np.where(s + v > 0, 1, 0).astype(np.uint8)  # Casting back into 8bit integer

        background = np.where(foreground == 0, 255, 0).astype(np.uint8)  # Invert foreground to get background in uint8
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
        foreground = cv2.bitwise_and(input_img, input_img,
                                     mask=foreground)  # Apply our foreground map to original image
        finalimage = background + foreground  # Combine foreground and background

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

    def normalize_img(self, img):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Image to normalize
        path : str
            Path to save the image

        Returns
        -------
        Image : np.ndarray 3-dim
            Normalized image
        """
        return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def get_black_and_white_img(self, img):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Image to convert to black and white

        Returns
        -------
        Image : np.ndarray 3-dim
            Black and white image
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    def resize_image(self, img, width=20, height=20):
        """
        Parameters
        ----------
        img : np.ndarray 3-dim
            Image to resize
        width : int
            Width of the image
        height : int
            Height of the image

        Returns
        -------
        Image : np.ndarray 3-dim
            Resized image
        """
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

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
