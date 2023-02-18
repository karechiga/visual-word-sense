"""
    main.py
    Trains a model to match a word with a given image out of a set of images.
    Evaluates performance of a given model on a given dataset of words and images.
"""

import argparse
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
import os, psutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING GPU")
class Model(torch.nn.Module):

    def __init__(self, data_dir, img_path):
        super(Model, self).__init__()
        self.img_path = img_path
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
        # self.conv = torch.nn.Conv2d(5,5,2)
        # self.pooling = torch.nn.MaxPool2d(2)
        
    def image_model(self, images):
        # Returns outputs of NNs with input of multiple images
        samples = []
        for n, row in enumerate(images):
            seqs = []
            for i in row:
                img = Image.open(self.img_path + i).convert('RGB')
                batch = self.i_preprocess(img).unsqueeze(0).to(device)
                x = self.i_pretrained(batch).to(device)
                seqs.append(x)
            samples.append(torch.stack(seqs).to(device))
        return torch.stack(samples).to(device)
    def word_model(self, words):
        # Returns the output of a NN with an input of two words
        x = self.w_embeddings(words.to(device))
        x = self.linear1(x).to(device)
        x = self.tanh(x).to(device)
        x = self.linear2(x).to(device)
        x = self.tanh(x).to(device)
        return x
    def forward(self, words, images):
        # Networks to be used for the images of the dataset
        i_seqs = self.image_model(images)
        # print("images pretrained: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        # Network to be used for the words of the dataset
        w_seq = self.word_model(words)
        # print("words and images pretrained: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        # Combining the word and image tensors
        out_tensor = []
        for i, sample in enumerate(i_seqs):
            samps = []
            for img in sample:
                combined = torch.cat((img, w_seq[i])).to(device)
                samps.append(combined.flatten().to(device))
            out_tensor.append(torch.stack(samps).to(device))
        linear = torch.nn.Linear(out_tensor[0].size(1), 1).to(device)
        out = linear(torch.stack(out_tensor).to(device)).squeeze(2).to(device)
        out = self.softmax(out).to(device)
        # print("forward output: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        return out

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

def tokenize(words, data_dir):
    # takes in a 2D list of words and returns a 2D list of their corresponding tokens.
    tokens = emb.getTokenizedVocab(data_dir)
    out = np.zeros(shape=(len(words), len(words[0])))
    for i in range(len(words)):
        for j, word in enumerate(words[i]):
            try:
                out[i,j] = tokens[word]
            except:
                out[i,j] = 0    # token is zero for unknown tokens
    return out

def untokenize(tokens, data_dir):
    vocab = emb.getTokenizedVocab(data_dir)
    key_list = list(vocab.keys())
    val_list = list(vocab.values())
    key_list[val_list.index(100)]
    out = []
    for i in range(len(tokens)):
        row = []
        for j, token in enumerate(tokens[i]):
            row.append(key_list[val_list.index(eval(token))])
        out.append(row)
    return out

def preprocessData(data_dir):
    # Read the training data from "data_dir"
    # Reading the text data
    X = readXData(glob.glob(data_dir + '/*train*/*data*')[0])
    tokens = tokenize([t[0:2] for t in X], data_dir)
    for i, row in enumerate(X):
        X[i][0:2] = tokens[i]
    y = readYData(glob.glob(data_dir + '/*train*/*gold*')[0])
    # split the rows into training and dev data randomly
    x_train, x_dev, y_train, y_dev = train_test_split(X,y,
                                   random_state=162,
                                   test_size=0.2,
                                   shuffle=True)
    # for development purposes, cut the number of samples to 100
    x_train = x_train[:100]
    x_dev = x_dev[:100]
    y_train = y_train[:100]
    y_dev = y_dev[:100]
    ###########################################################
    return np.vstack((np.array(x_train).T,np.array(y_train))).T, np.vstack((np.array(x_dev).T,np.array(y_dev))).T

def epoch(e, batches, model, loss_fnc, optimizer):
    running_loss = 0.
    print("Start of epoch {}: {} GB".format(e+1, psutil.Process(os.getpid()).memory_info().rss/1000000000))
    for i, data in enumerate(batches):
        print('EPOCH {}: Batch {} out of {}'.format(e + 1, i+1, len(batches)))
        # Every data instance is an input + label pair
        words, images, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # Make predictions for this batch
        outputs = model.forward(words, images)
        # predictions = torch.argmax(outputs, dim=1)
        actual = torch.Tensor([np.where(images[n] == labels[n])[0][0] for n in range(len(images))]).to(device)
        # argmax function on the outputs
        # Compute the loss and its gradients
        loss = loss_fnc(outputs, actual.type(torch.LongTensor).to(device)).to(device)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        print('EPOCH {}: Batch {} out of {}: Running Loss is {}'.format(e + 1, i+1, len(batches), running_loss))
        # print("{} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))

    return running_loss / len(batches)

def getBatches(data, batch_size):
    batches = [] # the training data split into batches
    np.random.shuffle(data)
    # untokenize([x[0:2] for x in data], '../data')
    for i in range(0, len(data), batch_size):
        batch = []
        # Words input
        batch.append(torch.tensor([[eval(x[0]), eval(x[1])] for x in data[i:i+batch_size]]).to(torch.int64))
        # Images input
        batch.append([x[2:-1] for x in data[i:i+batch_size]])
        # output
        batch.append([x[-1] for x in data[i:i+batch_size]])
        batches.append(batch)
    return batches

def train(model_dir, data_dir, epochs=10, batch_size=10, learning_rate = 0.01):
    """
    Train the model on the given training dataset located at "data_dir".
    """
    train_data, dev_data = preprocessData(data_dir)
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = Model(data_dir, img_path).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_vloss = 1_000_000.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for e in range(epochs):
        batches = getBatches(train_data, batch_size)
        vbatches = getBatches(dev_data, batch_size) # validation data split into batches
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = epoch(e, batches, model, loss_fnc, optimizer)
        print("Epoch complete: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
        # We don't need gradients on to do reporting
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(vbatches):
            print("Validation loop {}".format(i+1))
            outputs = None
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            vwords, vimages, vlabels = vdata
            outputs = model.forward(vwords, vimages).to(device)
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # predictions = torch.argmax(outputs, dim=1)
            actual = torch.Tensor([np.where(vimages[n] == vlabels[n])[0][0] for n in range(len(vimages))]).to(device)
            # argmax function on the outputs
            vloss = float(loss_fnc(outputs, actual.type(torch.LongTensor).to(device)))
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # print("End of epoch {}: {} GB".format(e+1, psutil.Process(os.getpid()).memory_info().rss/1000000000))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
            torch.save(model.state_dict(), model_path)
    return


def predict(index, model, words, tokens, images, labels):
    outputs = model.forward(tokens, images)
    predictions = torch.argmax(outputs, dim=1)
    actual = torch.Tensor([np.where(images[n] == labels[n])[0][0] for n in range(len(images))])
    correct = 0
    for i, p in enumerate(predictions):
        if predictions[i] == actual[i]:
            correct += 1
            print("CORRECT:\t{}. {} {}\t| Predicted: ({}) {}\t| Actual: ({}) {}".format(
                    index + i, words[i][0], words[i][1], p, images[i][p], int(actual[i]), labels[i]))
        else:
            print("INCORRECT:\t{}. {} {}\t| Predicted: ({}) {}\t| Actual: ({}) {}".format(
                    index + i, words[i][0], words[i][1], p, images[i][p], int(actual[i]), labels[i]))
    return correct

def evaluate_dataset(data_dir, data, model, step_size):
    train_words = untokenize([x[0:2] for x in data], data_dir)
    correct = 0
    for i in range(0, len(data), step_size):
        # Words input
        tokens = torch.tensor([[eval(x[0]), eval(x[1])] for x in data[i:i+step_size]]).to(torch.int64)
        words = train_words[i:i+step_size]
        # Images input
        images = [x[2:-1] for x in data[i:i+step_size]]
        # output
        labels = [x[-1] for x in data[i:i+step_size]]
        correct += predict(i, model, words, tokens, images, labels)
    return correct

def evaluate(model_dir, data_dir, test_dir, step_size):
    if test_dir is not None:
        img_path = glob.glob(test_dir + '/*images*/')[0]
        X = readXData(glob.glob(test_dir + '/*data*')[0])
        tokens = tokenize([t[0:2] for t in X], data_dir)
        for i, row in enumerate(X):
            X[i][0:2] = tokens[i]
        y = readYData(glob.glob(test_dir + '/*gold*')[0])
    else:
        train_data, dev_data = preprocessData(data_dir)
        img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
        model = Model(data_dir, img_path).to(device)
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        # Evaluate on the training data
        print("Training Data evaluation")
        tcorrect = evaluate_dataset(data_dir, train_data, model, step_size)
        # Evaluate on the dev set
        print("Development Data evaluation")
        dcorrect = evaluate_dataset(data_dir, dev_data, model, step_size)
        print("Training Data: {} correct predictions out of {} ({} %)\n".format(tcorrect, len(train_data), 100*tcorrect/len(train_data)) +
              "Development Data: {} correct predictions out of {} ({} %)".format(dcorrect, len(dev_data), 100*dcorrect/len(dev_data)))
    return



if __name__ == "__main__":
    # sets up a command-line interface for "train" and "predict"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    predict_parser = subparsers.add_parser("evaluate")
    predict_parser.set_defaults(func=evaluate)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("data_dir")
    predict_parser.add_argument("--test_dir", type=str, default=None)
    predict_parser.add_argument("--step_size", type=int, default=2)
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)