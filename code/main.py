"""
    main.py
    Trains a model to match a word with a given image out of a set of images.
    Evaluates performance of a given model on a given dataset of words and images.
"""

import argparse
import os
import numpy as np
import cv2 as cv
import glob
import json
import re
from sklearn.model_selection import train_test_split

def createEmbeddingFile(data_dir):
    vocab = dict()
    with open(data_dir + '/embeddings/glove.6B.50d.txt','rt', encoding="utf8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab[i_word] = i_embeddings
    with open(data_dir + "/embeddings/vocab.json", "w") as outfile:
        outfile.write(json.dumps(vocab, indent=4))
    return

def readXData(data_dir):
    rows = []
    with open(data_dir,'rt', encoding="utf8") as fi:
        data = fi.read().split('\n')
    for i, r in enumerate(data):
        row = re.split('\t| ', r)
        row.pop(0)
        rows.append(row)
    return rows

def readYData(data_dir):
    with open(data_dir,'rt', encoding="utf8") as fi:
        data = fi.read().split('\n')
    return data

def train(model_dir, data_dir, epochs=10, batch_size=32, learning_rate = 0.1):
    # Read the training data from "data_dir"
    # Create a dictionary of images
    imgs = []
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    for img in os.listdir(img_path):
        imgs.append(img)
        # imgs[img] = cv.imread(img_path + img)
    # Create a json dictionary of pretrained word embeddings (if it doesn't already exist)
    if not os.path.exists(data_dir + "/embeddings/vocab.json"):
        createEmbeddingFile(data_dir)
    # to load the json file: json.load(open(data_dir + "/embeddings/vocab.json",'r'))
    # Reading the text data
    X = readXData(glob.glob(data_dir + '/*train*/*data*')[0])
    y = readYData(glob.glob(data_dir + '/*train*/*gold*')[0])
    # split the rows into training and dev data randomly
    x_train, x_dev, y_train, y_dev = train_test_split(X,y,
                                   random_state=162,
                                   test_size=0.2,
                                   shuffle=True)
    x_train
    return


def predict(model_dir, data_dir, out_dir, reference_dir):
    return



if __name__ == "__main__":
    # sets up a command-line interface for "train" and "predict"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    data_dir = '../data/train'
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=0.1)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("data_dir")
    predict_parser.add_argument("out_dir")
    predict_parser.add_argument("--evaluate", dest='reference_dir')
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)