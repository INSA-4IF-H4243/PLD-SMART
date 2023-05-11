import cv2
import sys

def subtractor(substractor, parameters):
        if substractor== "GMG": # slowest
            return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120,
                                                            decisionThreshold=0.8)
        if substractor== "MOG": # slow
            return cv2.bgsegm.createBackgroundSubtractorMOG(history=200,
                                                            nmixtures=5,
                                                            backgroundRatio=0.7,
                                                            noiseSigma=0)
        if substractor== "MOG2": # fast
            return cv2.createBackgroundSubtractorMOG2(history=parameters["history"], # default: 500
                                                      varThreshold=parameters["threshold"], # default: 100
                                                      detectShadows=True)
        if substractor== "KNN": # fast
            return cv2.createBackgroundSubtractorKNN(history=parameters["history"], # default: 500
                                                     dist2Threshold=parameters["threshold"], # default: 400
                                                     detectShadows=True)
        if substractor== "CNT": # fastest / unstable
            return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,
                                                            maxPixelStability=15*60,
                                                            useHistory=True,
                                                            isParallel=True)
        else:
            print("Invalid substractor")
            sys.exit(0)

def filter(image, filter, parameters=""):
        
        if filter== "closing": # dilation followed by erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=10)
        if filter== "opening": # erosion followed by dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        if filter== "dilation":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            return cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=3)
        if filter== "erosion":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, parameters["shape"])
            return cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=parameters["iterations"])
        else:
            print("Invalid filter")
            sys.exit(0)