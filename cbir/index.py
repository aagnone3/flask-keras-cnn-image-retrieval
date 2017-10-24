# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.linalg import norm

import argparse
import logging
try:
    import cPickle as pickle
except:
    import pickle
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-directory", required = True,
        help = "Name of directory which contains images to be indexed")
    ap.add_argument("-index", required = True,
        help = "Name of index file")
    return ap.parse_args()


def get_image_fns(path):
    '''
     Returns a list of filenames for all jpg images in a directory. 
    '''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


def extract_feat(img_path):
    '''
     Use vgg16 model to extract features
     Output normalized feature vector
    '''
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    
    input_shape = (224, 224, 3)
    model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)
        
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = model.predict(img)
    norm_feat = feat[0]/norm(feat[0])
    return norm_feat


if __name__ == "__main__":
    '''
     Extract features and index the images
    '''

    args = parse_args()
    img_list = get_image_fns(args.directory)
    
    logger.info("Extracting embeddings from {} images in {}".format(len(img_list), args.directory))
    
    feats = []
    names = []
    for i, img_path in enumerate(img_list):
        norm_feat = extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(os.path.join(args.directory, img_name))
        logger.info("Extracting embedding from image {}/{}".format(i, len(img_list)))
    feats = np.array(feats)

    # directory for storing extracted features
    output = args.index
    
    logger.info("Writing extracted embeddings to disk.")
    with open(output, 'wb') as fp:
        pickle.dump({'features': feats, 'names': names}, fp)

