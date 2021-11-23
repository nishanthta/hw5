import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    blur = skimage.filters.gaussian(skimage.color.rgb2gray(image), sigma = 1)
    thresh = skimage.filters.threshold_otsu(blur)
    bw = blur < thresh
    labels,_ = skimage.measure.label(bw, background=0, return_num = True, connectivity = 2)
    regions = skimage.measure.regionprops(labels)
    for i in range(len(regions)):
        if regions[i].area >= 500:
            bboxes.append(regions[i].bbox)
    return bboxes, bw