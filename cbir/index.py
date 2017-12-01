# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.linalg import norm
import argparse
import logging
import multiprocessing as mp
try:
    import cPickle as pickle
except:
    import pickle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
input_shape = (224, 224, 3)
model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)


def extract_feat(img_path):
    """
     Use vgg16 model to extract features
     Output normalized feature vector
    """
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    img = img_to_array(load_img(img_path, target_size=(input_shape[0], input_shape[1])))
    img = preprocess_input(np.expand_dims(img, axis=0))
    feat = model.predict(img)
    norm_feat = feat[0] / norm(feat[0])
    return norm_feat


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-directory", required = True,
        help = "Name of directory which contains images to be indexed")
    ap.add_argument("-index", required = True,
        help = "Name of index file")
    return ap.parse_args()


if __name__ == "__main__":
    """
     Extract features and index the images
    """

    args = parse_args()

    with open(args.fn) as fp:
        img_list = list(map(lambda line: line.strip("\n"), fp.readlines()))
    
    feats = []
    for i, img_path in enumerate(img_list):
        logger.info("Extracting embedding from image {}/{}".format(i, len(img_list)))
        feats.append(extract_feat(img_path))
    feats = np.array(feats)

    logger.info("Writing extracted embeddings to disk.")
    with open(args.index, 'wb') as fp:
        pickle.dump({'features': feats, 'names': img_list}, fp)

