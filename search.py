# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from index import extract_feat
try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", dest="example", required=True,
        help = "Name of directory which contains images to be indexed.")
    parser.add_argument("-i", "--index", dest="index", required=True,
        help = "Path to index of embeddings for known images.")
    parser.add_argument("-r", "--result", dest="result", required=True,
        help = "Path for output retrieved images.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # read in indexed images' feature vectors and corresponding image names
    logger.info("Loading index of embeddings.")
    with open(args.index, 'rb') as fp:
        index_dict = pickle.load(fp)
        feats = index_dict["features"]
        img_fns = index_dict["names"]
            
    logger.info("Searching the index for potential matches.")
        
    # extract query image's feature
    query = extract_feat(args.example)

    # compute sorted similarity scores for all members of the index
    similarities = np.dot(query, feats.T)
    ranked_inds = np.argsort(similarities)[::-1]
    ranked_scores = similarities[ranked_inds]

    # number of top retrieved images to show
    top_n = 3
    top_matches = [img_fns[index] for index in ranked_inds[0:top_n]]
    logger.info("Top {} images: {}".format(top_n, top_matches))
     

    # show the query image and the top 3 matches
    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.title("Query image.")
    plt.imshow(mpimg.imread(args.example))
    for i, fn in enumerate(top_matches):
        plt.subplot(4, 1, i + 2)
        plt.imshow(mpimg.imread(fn))

    plt.figure(2)
    plt.stem(ranked_scores)
    plt.grid(True)
    plt.title("Index similarity scores")
    plt.xlabel("Index Item")
    plt.ylabel("Similarity Score")

    plt.show()

