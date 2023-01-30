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
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image

class Model(torch.nn.Module):

    def __init__(self, data_dir):
        super(Model, self).__init__()
        self.img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
        self.w_embeddings = emb.wordEmbeddingLayer(data_dir)
        i_weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.i_pretrained = models.resnet50(weights=i_weights)
        self.i_pretrained.train()
        self.i_preprocess = i_weights.transforms()
        self.linear1 = torch.nn.Linear(50, 200)
        self.linear2 = torch.nn.Linear(200, 1000)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv = torch.nn.Conv2d(5,5,2)
        self.pooling = torch.nn.MaxPool2d(2)
        
    def image_model(self, images):
        # Returns outputs of NNs with input of multiple images
        samples = []
        for row in images:
            seqs = []
            for i in row:
                img = Image.open(self.img_path + i).convert('RGB')
                batch = self.i_preprocess(img).unsqueeze(0)
                x = self.i_pretrained(batch)
                seqs.append(x)
            samples.append(torch.stack(seqs))
        return torch.stack(samples)
    def word_model(self, words):
        # Returns the output of a NN with an input of two words
        x = self.w_embeddings(words)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x
    def forward(self, words, images):
        # Networks to be used for the images of the dataset
        i_seqs = self.image_model(images)
        # Network to be used for the words of the dataset
        w_seq = self.word_model(words)
        out_tensor = []
        # Combining the word and image tensors
        for seq in i_seqs:
            combined = torch.cat((seq.view(seq.size(0), -1),
                            w_seq.view(w_seq.size(0), -1)), dim=1)
            out_tensor.append(combined)
        out = self.softmax(out_tensor)
        return out


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

def tokenize(words, data_dir):
    # takes in a list of words and returns a list of their corresponding tokens.
    tokens = emb.getTokenizedVocab(data_dir)
    out = np.zeros(shape=(len(words), len(words[0])))
    for i in range(len(words)):
        for j, word in enumerate(words[i]):
            try:
                out[i,j] = tokens[word]
            except:
                out[i,j] = 0    # token is zero for unknown tokens
    return out

def preprocessData(data_dir):
    # Read the training data from "data_dir"
    # Reading the text data
    X = readXData(glob.glob(data_dir + '/*train*/*data*')[0])
    y = readYData(glob.glob(data_dir + '/*train*/*gold*')[0])
    # split the rows into training and dev data randomly
    x_train, x_dev, y_train, y_dev = train_test_split(X,y,
                                   random_state=162,
                                   test_size=0.2,
                                   shuffle=True)
    # for development purposes, cut the number of samples to 100
    x_train = x_train[:15]
    x_dev = x_dev[:15]
    y_train = y_train[:15]
    y_dev = y_dev[:15]
    ###########################################################
    return x_train, x_dev, y_train, y_dev

def train(model_dir, data_dir, epochs=10, batch_size=32, learning_rate = 0.1):
    """
    Train the model on the given training dataset located at "data_dir".
    """
    x_train, x_dev, y_train, y_dev = preprocessData(data_dir)
    x_train_w = torch.tensor(tokenize([x[0:2] for x in x_train], data_dir)).to(torch.int64)
    x_train_i = [x[2:] for x in x_train]
    x_dev_w = torch.tensor(tokenize([x[0:2] for x in x_dev], data_dir)).to(torch.int64)
    x_dev_i = [x[2:] for x in x_dev]
    model = Model(data_dir)
    model.forward(x_train_w, x_train_i)
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