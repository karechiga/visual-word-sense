import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def readXData(data_dir):
    rows = []
    with open(data_dir,'rt', encoding="utf8") as fi:
        data = fi.read().split('\n')
    for i, r in enumerate(data):
        row = re.split('\t| ', r)
        if len(row) < 10:
            continue
        row.pop(0)
        rows.append(row)
    return rows

def readYData(data_dir):
    with open(data_dir,'rt', encoding="utf8") as fi:
        data = fi.read().split('\n')
    return data[0:-1]

def preprocessData(data_dir, train_size, dev_size, rand = 162):
    # Read the training data from "data_dir"
    # Reading the text data
    X = readXData(glob.glob(data_dir + '/*train*/*data*')[0])
    tokens = emb.tokenize([t[0:2] for t in X], data_dir)
    for i, row in enumerate(X):
        X[i][0:2] = tokens[i]
    y = readYData(glob.glob(data_dir + '/*train*/*gold*')[0])
    # split the rows into training and dev data randomly
    x_train, x_dev, y_train, y_dev = train_test_split(X,y,
                                   random_state=rand,
                                   test_size=0.2,
                                   shuffle=True)
    # for development purposes, cut the number of samples
    # remove '<unk>' tokens
    train_idx = [x_train.index(x) for x in x_train if x[0] != 0 and x[1] != 0]
    dev_idx = [x_dev.index(x) for x in x_dev if x[0] != 0 and x[1] != 0]
    print("TRAINING DATA: Removing {} samples that include unknown tokens out of {} total samples.".format(
        len(y_train)-len(train_idx), len(y_train)))
    print("DEV DATA: Removing {} samples that include unknown tokens out of {} total samples.".format(
        len(y_dev)-len(dev_idx), len(y_dev)))
    x_train = [x_train[i] for i in train_idx]
    x_dev = [x_dev[i] for i in dev_idx]
    y_train = [y_train[i] for i in train_idx]
    y_dev = [y_dev[i] for i in dev_idx]
    if train_size < len(y_train):
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        print("Reducing Training data to {} elements".format(
            train_size))
    if dev_size < len(y_dev):
        x_dev = x_dev[:dev_size]
        y_dev = y_dev[:dev_size]
        print("Reducing Dev data to {} elements.".format(
            dev_size))
    ###########################################################
    return np.vstack((np.array(x_train).T,np.array(y_train))).T, np.vstack((np.array(x_dev).T,np.array(y_dev))).T

def getBatches(data, batch_size, rand = True, words = None):
    batches = [] # the training data split into batches
    if words is not None:
        d = np.hstack([data,words])
    else:
        d = data
    if rand == True:
        np.random.shuffle(d)
    # separate words out of data
    if words is not None:
        data = d[:, :13]
        words = d[:, 13:]
    for i in range(0, len(data), batch_size):
        batch = []
        # Token input
        batch.append(torch.tensor([[eval(x[0]), eval(x[1])] for x in data[i:i+batch_size]]).to(torch.int64))
        # Images input
        batch.append([x[2:-1] for x in data[i:i+batch_size]])
        # output
        batch.append([x[-1] for x in data[i:i+batch_size]])
        # Optionally, add the actual words to the batch
        if words is not None:
            batch.append(words[i:i+batch_size])
        else:
            batch.append(None)
        batches.append(batch)
    return batches