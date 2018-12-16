#!/usr/bin/env python

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os, sys
import cv2


# The path for annotation file
annFile = '../../annotations/instances_train2017.json'

# The save path for JPEG image
JPEGImagesPath = '/home/jovyan/Projects/COCO17_Person/JPEGImages'

# The save path for mask PNG image
SegmentationClassPath = '/home/jovyan/Projects/COCO17_Person/SegmentationClass'


# Initialize COCO api for instance annotations
coco=COCO(annFile)


# Get all images containing given 'person' category
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
print("Total number of images: {}".format(len(imgIds)))

images = coco.loadImgs(imgIds)
for img in images:
    filename = img['coco_url'].split('/')[-1]

    if os.path.isfile(JPEGImagesPath + '/' + filename):
        continue

    # Use url to load image and save it
    I = io.imread(img['coco_url'])
    io.imsave(JPEGImagesPath + '/' + filename, I)

    # Get annotation of this image
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    # Generate segmentation mask and save it
    mask = np.zeros(I.shape[:2], dtype=np.uint8)    
    for ann in anns:
        mask = (mask | coco.annToMask(ann))

    count = np.sum(mask)
    mask = np.stack([mask * 0, mask * 0, mask * 128], axis=2)

    cv2.imwrite(SegmentationClassPath + '/' + os.path.splitext(filename)[0] + '.png', mask)

    info = [os.path.splitext(filename)[0], I.shape[0], I.shape[1], len(anns), count / (I.shape[0] * I.shape[1]) * 100]
    print(','.join(list(map(str, info))))
    sys.stdout.flush()

