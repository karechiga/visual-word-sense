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
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING GPU")
class Model(torch.nn.Module):

    def __init__(self, data_dir, img_path):
        super(Model, self).__init__()
        self.img_path = img_path
        self.w_embeddings = emb.wordEmbeddingLayer(data_dir)
        i_weights = models.ResNet50_Weights.IMAGENET1K_V2
        res50 = models.resnet50(weights=i_weights)
        layers = list(res50._modules.keys())
        # Want to remove the final linear layer of ResNet50
        self.i_pretrained = torch.nn.Sequential(*[res50._modules[x] for x in layers[:-1]])
        self.i_pretrained.train()
        self.i_preprocess = i_weights.transforms()
        res50 = None
        i_weights = None
        layers = None
        self.linear1 = torch.nn.Linear(300, 512, device=device)
        self.linear2 = torch.nn.Linear(512, 2048, device=device)
        self.linear3 = torch.nn.Linear(6144, 1, device=device)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax(dim=1)
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
                seqs.append(x.flatten())
            samples.append(torch.stack(seqs).to(device))
        return torch.stack(samples).to(device)
    def word_model(self, words):
        # Returns the output of a NN with an input of two words
        x = self.w_embeddings(words.to(device))
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
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
                combined = torch.cat((img.unsqueeze(0), w_seq[i])).to(device)
                samps.append(combined.flatten().to(device))
            out_tensor.append(torch.stack(samps).to(device))
        out = self.linear3(torch.stack(out_tensor).to(device)).squeeze(2)
        # out = self.softmax(out).to(device)
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

def plotPerformance(losses, accs):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.plot(np.arange(1,len(losses)+1), losses, 'r:')
    ax2.plot(np.arange(1,len(accs)+1), accs, 'b:')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim([0, 100])
    ax1.set_ylabel('Cross Entropy Loss')
    ax2.set_ylabel('Model Accuracy (%)')
    ax1.set_xticks(np.arange(1,len(losses)+1, 2))
    ax2.set_xticks(np.arange(1,len(accs)+1, 2))
    return ax1, ax2

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

def preprocessData(data_dir, train_size, dev_size):
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
    # for development purposes, cut the number of samples
    # remove '<unk>' tokens
    train_idx = [x_train.index(x) for x in x_train if x[0] == 0 or x[1] == 0]
    dev_idx = [x_dev.index(x) for x in x_dev if x[0] == 0 or x[1] == 0]
    print("TRAINING DATA: Removing {} samples that include unknown tokens out of {} total samples.".format(
        len(train_idx), len(y_train)-len(train_idx)))
    print("DEV DATA: Removing {} samples that include unknown tokens out of {} total samples.".format(
        len(dev_idx), len(y_dev)-len(dev_idx)))
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

def epoch(e, batches, model, loss_fnc, optimizer):
    running_loss = 0.
    correct = 0
    total = 0
    losses = []
    accuracies = []
    for i, data in enumerate(batches):
        print('EPOCH {}: Batch {} out of {}'.format(e + 1, i+1, len(batches)))
        # Every data instance is an input + label pair
        words, images, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = None
        outputs = model.forward(words, images)
        softmax = torch.nn.Softmax(dim=1)
        predictions = torch.argmax(softmax(outputs), dim=1)
        actual = torch.Tensor([np.where(images[n] == labels[n])[0][0] for n in range(len(images))]).to(device)
        correct += int(sum(predictions == actual))
        total += len(predictions)
        # argmax function on the outputs
        # Compute the loss and its gradients
        loss = loss_fnc(outputs, actual.type(torch.LongTensor).to(device)).to(device)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += float(loss.item())
        losses.append(round(running_loss/(i+1),2))
        accuracies.append(round(100*correct / total, 2))
        print('EPOCH {}: Batch {} out of {}: Average Loss is {}, Accuracy is {}%'.format(
            e + 1, i+1, len(batches), losses[i], accuracies[i]))
    return running_loss / len(batches), correct / total, losses, accuracies

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

def train(model_dir, data_dir, epochs=10, batch_size=10, learning_rate = 0.01, train_size=1000000, dev_size=1000000):
    """
    Train the model on the given training dataset located at "data_dir".
    """
    train_data, dev_data = preprocessData(data_dir, train_size, dev_size)
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = Model(data_dir, img_path).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    # loss_fnc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_vloss = 1_000_000.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    losses = []
    accs = []
    v_losses = []
    v_accs = []
    model_dir = model_dir + '/model_{}'.format(timestamp)
    os.mkdir(model_dir)
    for e in range(epochs):
        batches = getBatches(train_data, batch_size)
        vbatches = getBatches(dev_data, batch_size) # validation data split into batches
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = None
        accuracy = 0
        avg_loss, accuracy, e_losses, e_accs = epoch(e, batches, model, loss_fnc, optimizer)
        losses.append(round(avg_loss,2))
        accs.append(round(100*accuracy,2))
        # Plot the epoch and save it
        ax1, ax2 = plotPerformance(e_losses, e_accs)
        ax1.set_title('Epoch {}/{} LOSS: batch_size = {}, learning_rate = {}'.format(
            e+1, epochs, batch_size, learning_rate
        ))
        ax2.set_title('Epoch {}/{} ACCURACY: batch_size = {}, learning_rate = {}'.format(
            e+1, epochs, batch_size, learning_rate
        ))
        plt.xlabel('Batch Number')
        plt.savefig(model_dir + '/{}_Epoch{}_BS{}_LR{}.png'.format(
            timestamp, e+1, batch_size, learning_rate))
        plt.cla()
        e_accs = None
        e_losses = None
        # Clear the plot after saving it
        model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e+1)
        # We don't need gradients on to do reporting
        model.train(False)
        running_vloss = 0.0
        correct = 0
        total = 0
        for i, vdata in enumerate(vbatches):
            print("Validation loop {}".format(i+1))
            outputs = None
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            vwords, vimages, vlabels = vdata
            outputs = model.forward(vwords, vimages).to(device)
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            softmax = torch.nn.Softmax(dim=1)
            predictions = torch.argmax(softmax(outputs), dim=1)
            actual = torch.Tensor([np.where(vimages[n] == vlabels[n])[0][0] for n in range(len(vimages))]).to(device)
            correct += int(sum(predictions == actual))
            total += len(predictions)
            # argmax function on the outputs
            vloss = float(loss_fnc(outputs, actual.type(torch.LongTensor).to(device)))
            running_vloss += vloss
        outputs = None
        v_losses.append(round(running_vloss / (i + 1),2))
        v_accs.append(round(100 * correct / total,2))
        print('Loss:\t\ttrain {}\tvalidation {}'.format(losses[e], v_losses[e]))
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(accs[e], v_accs[e]))

        # Track best performance, and save the model's state
        if v_accs[e] > 50:  # Only save model if it is greater than 50% accuracy on the dev set
            print('saving model')
            model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
            torch.save(model.state_dict(), model_path)
    # Plot performance on training data
    ax1, ax2 = plotPerformance(losses, accs)
    ax1.set_title('Training {} LOSS: batch_size = {}, learning_rate = {}'.format(
        timestamp, batch_size, learning_rate
    ))
    ax2.set_title('Training {} ACCURACY: batch_size = {}, learning_rate = {}'.format(
        timestamp, batch_size, learning_rate
    ))
    plt.xlabel('Epoch Number')
    plt.savefig(model_dir + '/{}_TrainData_BS{}_LR{}.png'.format(
        timestamp, batch_size, learning_rate))
    plt.cla()
    # Plot Performance on Dev data
    ax1, ax2 = plotPerformance(v_losses, v_accs)
    ax1.set_title('Development {} LOSS: batch_size = {}, learning_rate = {}'.format(
        timestamp, batch_size, learning_rate
    ))
    ax2.set_title('Development {} ACCURACY: batch_size = {}, learning_rate = {}'.format(
        timestamp, batch_size, learning_rate
    ))
    plt.xlabel('Epoch Number')
    plt.savefig(model_dir + '/{}_DevData_BS{}_LR{}.png'.format(
        timestamp, batch_size, learning_rate))
    plt.cla()
    return


def predict(index, model, words, tokens, images, labels):
    outputs = model.forward(tokens, images)
    softmax = torch.nn.Softmax(dim=1)
    predictions = torch.argmax(softmax(outputs), dim=1)
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
    train_parser.add_argument("--train_size", type=int, default=1000000)
    train_parser.add_argument("--dev_size", type=int, default=1000000)
    predict_parser = subparsers.add_parser("evaluate")
    predict_parser.set_defaults(func=evaluate)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("data_dir")
    predict_parser.add_argument("--test_dir", type=str, default=None)
    predict_parser.add_argument("--step_size", type=int, default=2)
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)