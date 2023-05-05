import os
from processor import ImageProcessor
import cv2 as cv

from video import Image
image_processor = ImageProcessor()

# Load image
# Get absolute path of the image
# Bugging!
project_path = os.path.dirname(os.path.dirname(__file__))
print(project_path)
image_path = os.path.join(project_path, 'img', 'frame0-crop.jpg')
print(image_path)
image = Image.load_image(image_path)
imread_image = image.img_imread


image_bw = image_processor.get_black_and_white_img(imread_image)
image_normalized = image_processor.normalize_img(imread_image)

# Delete background of each image
image_no_bg = image_processor.remove_background(imread_image)
image_bw_no_bg = image_processor.remove_background(image_bw)
image_normalized_no_bg = image_processor.remove_background(image_normalized)


# Display all images using cv2
# The positions of the windows in the code are put in reverse, so the images with the background appear first.
cv.imshow('Normalized image without background', image_normalized_no_bg)
cv.imshow('Black and white image without background', image_bw_no_bg)
cv.imshow('Original image without background', image_no_bg)
cv.imshow('Normalized image', image_normalized)
cv.imshow('Black and white image', image_bw)
cv.imshow('Original image', imread_image)

cv.waitKey(0)
cv.destroyAllWindows()



