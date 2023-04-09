import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
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
    # Split images 80-20
    imgs = glob.glob(data_dir + '/*train*/*images*/*.jpg')
    imgs = [x[re.search('image\.(.*)',x).regs[0][0]:] for x in imgs]
    np.random.shuffle(imgs)
    train_img = imgs[:int(np.floor(len(imgs)*0.8))]
    dev_img = imgs[int(np.floor(len(imgs)*0.8)):]
    
    # Reading the text data
    X = readXData(glob.glob(data_dir + '/*train*/*data*')[0])
    y = readYData(glob.glob(data_dir + '/*train*/*gold*')[0])
    
    t_idx = [i for i in range(len(y)) if y[i] in train_img]
    d_idx = [i for i in range(len(y)) if y[i] in dev_img]
    np.random.shuffle(t_idx)
    np.random.shuffle(d_idx)
    y_train = [y[i] for i in t_idx]
    y_dev = [y[i] for i in d_idx]
    x_train = [X[i][0:2] for i in t_idx]
    x_dev = [X[i][0:2] for i in d_idx]
    # split the rows into training and dev data randomly
    # x_train, x_dev, y_train, y_dev = train_test_split(X,y,
    #                                random_state=rand,
    #                                test_size=0.2,
    #                                shuffle=True)
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
    return np.vstack((np.array(x_train).T,np.array(y_train))).T, np.vstack((np.array(x_dev).T,np.array(y_dev))).T, train_img, dev_img

def getBatches(data, images, batch_size, rand = True):
    batches = [] # the training data split into batches
    if rand == True:
        np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = []
        words = []
        options = []
        answers = []
        for x in data[i:i+batch_size]:
            # Word input
            words.append(x[0:2])
            # Images input
            imgs = np.random.choice([j for j in images if j != x[-1]],size=9,replace=False)
            imgs = np.hstack([imgs,x[-1]])
            np.random.shuffle(imgs)
            options.append(imgs)
            # output
            answers.append(x[-1])
        batch.append(words)
        batch.append(options)
        batch.append(answers)
        batches.append(batch)
    return batches