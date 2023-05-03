from rembg import remove, new_session
from ..video.Image import Image
from skimage.restoration import estimate_sigma


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

    def remove_background(self, input_path: str, threshold=0.3):
        input_obj = Image.load_image(input_path)
        input_img = input_obj.image
        model_name = "u2netp" if (estimate_noise(input_img) < threshold) else "u2net_human_seg"
        session = new_session(model_name=model_name)
        output = remove(input_img, session=session, post_process_mask=True)
        return output


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
