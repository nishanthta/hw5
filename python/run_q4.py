import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from collections import defaultdict
from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('./images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('./images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    # plt.show()
    # print(bboxes[0])
    bboxes_sorted = bboxes
    bboxes_sorted.sort()
    rows = defaultdict(list)
    curr_row = 0
    # print(bboxes_sorted[0])
    rows[curr_row].append(bboxes_sorted[0])
    for i in range(1, len(bboxes_sorted)):
        if (bboxes_sorted[i][0] - bboxes_sorted[i - 1][0]) > (bboxes_sorted[i - 1][2] - bboxes_sorted[i - 1][0]): #new row
            curr_row += 1
        rows[curr_row].append(bboxes_sorted[i])
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    all_letters = []
    for box in bboxes_sorted:
        x1, y1, x2, y2 = box
        cropped = bw[x1:x2 + 1, y1:y2 + 1]
        cropped_eroded = skimage.morphology.binary_erosion(cropped)
        cropped = skimage.transform.resize(cropped_eroded, (32, 32))
        all_letters.append(cropped.transpose().flatten())
    # continue
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    h1 = forward(all_letters, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)

    preds = letters[np.argmax(probs, axis = 1)]
    predrows = defaultdict(list)
    idx = 0
    for row, vals in rows.items():
        rowlen = len(vals)
        predrows[row] = preds[idx:idx + rowlen]
        idx += rowlen + 1




    continue
    