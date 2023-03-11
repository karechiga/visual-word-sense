"""
    main.py
    Trains a model to match a word with a given image out of a set of images.
    Evaluates performance of a given model on a given dataset of words and images.
"""

import argparse
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
import os
import wandb
import preprocess as pre

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING GPU")

def configureWandB(learning_rate, batch_size, epochs, dropout, train_size, dev_size,
                   word_linears, word_activations, out_linears, out_activations):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="visualwordsense",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": dropout,
        "train_size": train_size,
        "dev_size": dev_size,
        'word_linears': word_linears,
        'word_activations': word_activations,
        'out_linears': out_linears,
        'out_activations': out_activations
        }
    )

# def plotPerformance(losses, accs):
#     ax1 = plt.subplot(211)
#     ax2 = plt.subplot(212)
#     for i in range(len(losses)):
#         ax1.plot(np.arange(1,len(losses[i])+1), losses[i], ':')
#         ax1.set_xticks(np.arange(1,len(losses[0])+1, 2))
#     for i in range(len(accs)):
#         ax2.plot(np.arange(1,len(accs[i])+1), accs[i], ':')
#         ax2.set_xticks(np.arange(1,len(accs[0])+1, 2))
#     ax1.set_ylim(bottom=0)
#     ax2.set_ylim([0, 100])
#     ax1.set_ylabel('Cross Entropy Loss')
#     ax2.set_ylabel('Model Accuracy (%)')
#     return ax1, ax2

def train_epoch(e, batches, model, loss_fnc, optimizer):
    model.train(True)
    running_loss = 0.
    correct = 0
    total = 0
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
        print('out:\t{}'.format(outputs[0:3]))
        print('actual:\t{}'.format(actual[0:3]))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += float(loss.item())
        print('EPOCH {}: Batch {} out of {}: Average Loss is {}, Accuracy is {}%'.format(
            e + 1, i+1, len(batches), round(running_loss / (i+1),3), round(100*correct / total,2)))
    return running_loss / len(batches), correct / total

def eval_epoch(vbatches, model, loss_fnc):
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
        print('out:\t{}'.format(outputs[0:3]))
        print('actual:\t{}'.format(actual[0:3]))
        running_vloss += vloss
        print('Batch {} out of {}: Average Loss is {}, Accuracy is {}%'.format(
            i+1, len(vbatches), round(running_vloss / (i+1),3), round(100*correct / total,2)))
    outputs = None
    return running_vloss / (i + 1) , correct / total

def train(model_dir, data_dir, epochs=10, batch_size=10, learning_rate = 0.01, dropout=0.25, train_size=1000000, dev_size=1000000,
          word_linears = 2, word_activations = 'tanh', out_linears = 2, out_activations = 'relu'):
    """
    Train the model on the given training dataset located at "data_dir".
    """
    train_data, dev_data = pre.preprocessData(data_dir, train_size, dev_size)
    configureWandB(learning_rate, batch_size, epochs, dropout, len(train_data), len(dev_data),
                   word_linears, word_activations, out_linears, out_activations)
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
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
            w_activation = torch.nn.ReLU() if wandb.config['word_activations'] == 'relu' else torch.nn.Tanh()
            if wandb.config['word_linears'] == 1:
                self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 2048, device=device), w_activation)
            elif wandb.config['word_linears'] == 2:
                self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 512, device=device), w_activation,
                                                        torch.nn.Linear(512, 2048, device=device), w_activation)
            elif wandb.config['word_linears'] == 3:
                self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 512, device=device), w_activation,
                                                        torch.nn.Linear(512, 1024, device=device), w_activation,
                                                        torch.nn.Linear(1024, 2048, device=device), w_activation)
            
            o_activation = torch.nn.ReLU() if wandb.config['out_activations'] == 'relu' else torch.nn.Tanh()
            drop = torch.nn.Dropout(p=wandb.config['dropout'])
            if wandb.config['out_linears'] == 1:
                self.o_sequential = torch.nn.Sequential(drop, torch.nn.Linear(6144, 1, device=device))
            elif wandb.config['out_linears'] == 2:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                        o_activation, torch.nn.Linear(3072, 1, device=device))
            elif wandb.config['out_linears'] == 3:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                        o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                        o_activation, torch.nn.Linear(1024, 1, device=device))
            elif wandb.config['out_linears'] == 4:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                        o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                        o_activation, torch.nn.Linear(1024, 512, device=device),
                                                        drop, o_activation, torch.nn.Linear(512, 1, device=device))
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
            x = self.w_sequential(x)
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
                    x = torch.cat((img.unsqueeze(0), w_seq[i])).to(device)
                    samps.append(x.flatten())
                out_tensor.append(torch.stack(samps).to(device))
            out = self.o_sequential(torch.stack(out_tensor).to(device)).squeeze(2)
            # print("forward output: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
            return out

    model = Model(data_dir, img_path).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    # loss_fnc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_vloss = 1_000_000.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('{} - lr{} batchsize{} epochs{} train_size{} dev_size{}'.format(
        timestamp, learning_rate, batch_size, epochs, len(train_data), len(dev_data)))
    for e in range(epochs):
        batches = pre.getBatches(train_data, batch_size)
        vbatches = pre.getBatches(dev_data, batch_size) # validation data split into batches
        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = None
        accuracy = 0
        avg_loss, accuracy = train_epoch(e, batches, model, loss_fnc, optimizer)
        wandb.log({"train_acc": accuracy, "train_loss": avg_loss})
        model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e+1)
        val_loss = None
        val_acc = 0
        val_loss, val_acc = eval_epoch(vbatches, model, loss_fnc)
        wandb.log({"val_acc": val_acc, "val_loss": val_loss})
        print('Loss:\t\ttrain {}\tvalidation {}'.format(round(avg_loss,2), round(val_loss,2)))
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(round(100*accuracy,2), round(val_acc,2)))

        # Track best performance, and save the model's state
        if val_acc > 0.70:  # Only save model if it is greater than 50% accuracy on the dev set
            print('saving model')
            model_dir = model_dir + '/{}_lr{}_b{}_e{}_train{}'.format(
            timestamp, round(learning_rate, 7), batch_size, epochs, len(train_data))
            os.mkdir(model_dir)
            model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
            torch.save(model.state_dict(), model_path)
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
    train_words = emb.untokenize([x[0:2] for x in data], data_dir)
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
        X = pre.readXData(glob.glob(test_dir + '/*data*')[0])
        tokens = emb.tokenize([t[0:2] for t in X], data_dir)
        for i, row in enumerate(X):
            X[i][0:2] = tokens[i]
        y = pre.readYData(glob.glob(test_dir + '/*gold*')[0])
    else:
        train_data, dev_data = pre.preprocessData(data_dir)
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
    train_parser.add_argument("--dropout", type=float, default=0.25)
    train_parser.add_argument("--train_size", type=int, default=1000000)
    train_parser.add_argument("--dev_size", type=int, default=1000000)
    train_parser.add_argument("--word_linears", type=int, default=2)
    train_parser.add_argument("--word_activations", type=str, default='tanh')
    train_parser.add_argument("--out_linears", type=int, default=2)
    train_parser.add_argument("--out_activations", type=str, default='relu')
    predict_parser = subparsers.add_parser("evaluate")
    predict_parser.set_defaults(func=evaluate)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("data_dir")
    predict_parser.add_argument("--test_dir", type=str, default=None)
    predict_parser.add_argument("--step_size", type=int, default=2)
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)