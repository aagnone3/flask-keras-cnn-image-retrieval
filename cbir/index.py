# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import logging
import argparse
import numpy as np
import multiprocessing as mp
from numpy.linalg import norm

try:
    import cPickle as pickle
except ImportError:
    import pickle

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INPUT_SHAPE = (224, 224, 3)
# INPUT_SHAPE = (512, 512, 3)
model = VGG16(weights="imagenet",
              input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]),
              pooling="max",
              include_top=False)


def extract_feat(img_path):
    """
     Use vgg16 model to extract features
     Output normalized feature vector
    """
    img = img_to_array(load_img(img_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])))
    img = preprocess_input(np.expand_dims(img, axis=0))
    feat = model.predict(img)
    norm_feat = feat[0] / norm(feat[0])
    return norm_feat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_list", dest="file_list_fn", required=True,
                        help="Name of directory which contains images to be indexed")
    parser.add_argument("-o", "--output", dest="index_fn", required=True,
                        help="Name of index file")
    return parser.parse_args()


def index_features(args):
    """
    Extract features and index the images.
    :param args:
        Namespace arguments.
    :return:
        No return value.
    """

    with open(args.file_list_fn) as fp:
        img_list = list(map(lambda line: line.strip("\n"), fp.readlines()))

    feats = []
    for i, img_path in tqdm(enumerate(img_list), total=len(img_list)):
        feats.append(extract_feat(img_path))
    feats = np.array(feats)

    logger.info("Writing extracted embeddings to disk.")
    with open(args.index_fn, "wb") as fp:
        pickle.dump({"features": feats, "names": img_list}, fp)


if __name__ == "__main__":
    index_features(parse_args())
