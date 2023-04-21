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
        imgs = row[-10:]
        imgs.insert(0, ' '.join(row[:-10]))
        rows.append(imgs)
    return rows

def readYData(data_dir):
    with open(data_dir,'rt', encoding="utf8") as fi:
        data = fi.read().split('\n')
    return data[0:-1]

def preprocessData(data_dir, train_size, dev_size, shuffle_options = True):
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
    if shuffle_options:
        # Image options for each sample will be shuffled during batching
        x_train = [X[i][0:2] for i in t_idx]
        x_dev = [X[i][0:2] for i in d_idx]
    else:
        # Image options will remain the same for each sample
        x_train = [X[i] for i in t_idx]
        x_dev = [X[i] for i in d_idx]

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

def getBatches(data, images, batch_size, rand = True, shuffle_options = True):
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
            words.append(x[0])
            # Images input
            if shuffle_options:
                imgs = np.random.choice([j for j in images if j != x[-1]],size=9,replace=False)
                imgs = np.hstack([imgs,x[-1]])
                np.random.shuffle(imgs)
                options.append(imgs)
            else:
                # leave options unchanged from data input
                options.append(x[1:-1])
            # output
            answers.append(x[-1])
        batch.append(words)
        batch.append(options)
        batch.append(answers)
        batches.append(batch)
    return batches