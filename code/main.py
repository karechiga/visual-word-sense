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
import copy
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING GPU")

def configureWandB(learning_rate, batch_size, epochs, dropout, train_size, dev_size,
                   word_linears, word_activations, out_linears, out_activations, seed,
                   early_stop, model):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="visualwordsense",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_size": train_size,
            "dev_size": dev_size,
            'random_seed': seed,
            'early_stop': early_stop,
            'model_config': {
                'model': model,
                'word_linears': word_linears,
                'word_activations': word_activations,
                'out_linears': out_linears,
                "dropout": dropout,
                'out_activations': out_activations
            }
        }
    )

def run_batch(batch, model, training = False, loss_fnc = None, optimizer = None, sample_num = 0):
    model.train(training)
    if training == True:
        optimizer.zero_grad()
    tokens, images, labels, words = batch
    outputs = model.forward(tokens, images)
    softmax = torch.nn.Softmax(dim=1)
    predictions = torch.argmax(softmax(outputs), dim=1)
    actual = torch.Tensor([np.where(images[n] == labels[n])[0][0] for n in range(len(images))]).to(device)
    if words is not None:   # If including non-tokenized words in the batch, output them and their predictions.
        output_predictions(index=sample_num, words=words, images=images,
                           outputs=softmax(outputs).cpu().detach().numpy(), predictions=predictions,
                           answers=actual, labels=labels)
    correct = int(sum(predictions == actual))
    total = len(predictions)
    print('{} out of {} ({}%) correctly predicted:'.format(correct, total, round(100*correct/total,2)))
    l = 0
    if loss_fnc is not None:
        # Compute the loss
        loss = loss_fnc(outputs, actual.type(torch.LongTensor).to(device)).to(device)
        l = float(loss.item())
    if training == True:
        # Compute gradients
        loss.backward()
        # Adjust learning weights
        optimizer.step()
    return correct, total, l

def run_epoch(e, batches, model, training = False, loss_fnc = None, optimizer = None):
    running_loss = 0
    correct = 0
    total = 0
    string = 'TRAINING: ' if training == True else 'EVALUATION: '
    for i, batch in enumerate(batches):
        print(string + 'EPOCH {}: Batch {} out of {}'.format(e + 1, i+1, len(batches)))
        c, t, l = run_batch(batch, model, training=training, loss_fnc=loss_fnc, optimizer=optimizer, sample_num=total)
        # Gather data and report
        correct += c
        total += t
        running_loss += l
        if loss_fnc is not None:
            print('Average Loss is {}, Overall Accuracy is {}/{} ({}%)\n'.format(
                round(running_loss / (i+1),3), correct, total, round(100*correct / total,2)))
        else:
            print('Overall Accuracy is {}/{} ({}%)\n'.format(
                correct, total, round(100*correct / total,2)))
    return running_loss / len(batches), correct/total

def train(model_dir, data_dir, train_data, dev_data):
    """
    Train the model on the given training dataset located at "data_dir".
    """

    np.random.seed(wandb.config.random_seed)

    if wandb.config.model_config['model'] == 'model1':
        import model1 as md
        print('Using model1.py for the model.')
    elif wandb.config.model_config['model'] == 'model2':
        import model2 as md
        print('Using model2.py for the model.')
    else:
        import model as md
        print('Using model.py for the model.')

    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = md.Model(data_dir, img_path, wandb.config.model_config).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    best_v_acc = 0.0; best_num_epochs = 0; count = 0
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('{} - lr{} batchsize{} epochs{} train_size{} dev_size{}'.format(
        timestamp, wandb.config.learning_rate, wandb.config.batch_size, wandb.config.epochs, wandb.config.train_size, wandb.config.dev_size))
    for e in range(wandb.config.epochs):
        batches = pre.getBatches(train_data, wandb.config.batch_size)
        vbatches = pre.getBatches(dev_data, wandb.config.batch_size) # validation data split into batches
        # Make sure gradient tracking is on, and do a pass over the data
        ##############################################################################################################################
        # For displaying words and their predictions in the first and last batch of the epoch
        batches[0][3] = emb.untokenize([list(map(str, x[0:2].cpu().detach().numpy())) for x in batches[0][0]], data_dir)
        batches[-1][3] = emb.untokenize([list(map(str, x[0:2].cpu().detach().numpy())) for x in batches[-1][0]], data_dir)
        ##############################################################################################################################
        avg_loss = None; accuracy = 0
        avg_loss, accuracy = run_epoch(e=e, batches=batches, model=model, training=True, loss_fnc=loss_fnc, optimizer=optimizer)
        val_loss = None; val_acc = 0
        val_loss, val_acc = run_epoch(e=e, batches=vbatches, model=model, loss_fnc=loss_fnc)
        wandb.log({"epoch": e, "train_acc": accuracy, "train_loss": avg_loss, "val_acc": val_acc, "val_loss": val_loss})
        print('Loss:\t\ttrain {}\tvalidation {}'.format(round(avg_loss,2), round(val_loss,2)))
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(round(100*accuracy,2), round(100*val_acc,2)))

        # early stopping
        if val_acc > best_v_acc:
            best_v_acc = val_acc
            best_num_epochs = e
            best_model = None
            best_model = copy.deepcopy(model.to('cpu').state_dict())
            model.to(device)
            count = 0
        else:
            count += 1
            if count >= wandb.config.early_stop:
                print("Early stopping initiated after {} consecutive epochs below the best validation accuracy.".format(count))
                break
    wandb.log({"best_val_acc": best_v_acc, "best_num_epochs": best_num_epochs})

    print('saving model')
    model_dir = model_dir + '/{}_lr{}_b{}_e{}_train{}'.format(
    timestamp, round(wandb.config.learning_rate, 7), wandb.config.batch_size, best_num_epochs, wandb.config.train_size)
    os.mkdir(model_dir)
    print('Model dir: {}'.format(model_dir))
    model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
    config_path = model_dir + '/' + 'config_{}_{}.json'.format(timestamp, e)
    torch.save(best_model, model_path)
    with open(config_path, "w") as outfile:
        outfile.write(json.dumps(wandb.config.model_config, indent=4))
    return


def output_predictions(index, words, images, outputs, predictions, answers, labels):
    for i, p in enumerate(predictions):
        if predictions[i] == answers[i]:
            print("CORRECT:\t{}. {} {}\t| Predicted: ({}) {}\t| Answer: ({}) {}".format(
                    index + i, words[i][0], words[i][1], p, images[i][p], int(answers[i]), labels[i]))
        else:
            print("INCORRECT:\t{}. {} {}\t| Predicted: ({}) {}\t| Answer: ({}) {}".format(
                    index + i, words[i][0], words[i][1], p, images[i][p], int(answers[i]), labels[i]))
        top_3 = outputs[i].argsort()[-3:][::-1]
        c = ['*' if t == int(answers[i]) else '' for t in top_3]
        print("Top 3:\t1. ({}) {}{}, {}%\t2. ({}) {}{}, {}%\t3. ({}) {}{}, {}%\n".format(
                top_3[0], images[i][top_3[0]], c[0], round(100*outputs[i][top_3[0]],2),
                top_3[1], images[i][top_3[1]], c[1], round(100*outputs[i][top_3[1]],2),
                top_3[2], images[i][top_3[2]], c[2], round(100*outputs[i][top_3[2]],2)))
    return

def evaluate_dataset(data_dir, data, model, step_size):
    words = emb.untokenize([x[0:2] for x in data], data_dir)
    batches = pre.getBatches(data, step_size, rand=False, words=words)
    loss, acc = run_epoch(0, batches, model, training = False, loss_fnc = None, optimizer = None)
    return loss, acc

def evaluate(model_dir, data_dir, test_dir, step_size, train_size, dev_size):
    model_config = json.load(open(glob.glob(model_dir + '/*config*')[0],'r'))
    if model_config['model'] == 'model1':
        import model1 as md
        print('Using model1.py for the evaluation.')
    elif model_config['model'] == 'model2':
        import model2 as md
        print('Using model2.py for the evaluation.')
    else:
        import model as md
        print('Using model.py for the evaluation.')

    if test_dir is not None:
        img_path = glob.glob(test_dir + '/*images*/')[0]
        X = pre.readXData(glob.glob(test_dir + '/*data*')[0])
        tokens = emb.tokenize([t[0:2] for t in X], data_dir)
        for i, row in enumerate(X):
            X[i][0:2] = tokens[i]
        y = pre.readYData(glob.glob(test_dir + '/*gold*')[0])
    else:
        train_data, dev_data = pre.preprocessData(data_dir, train_size, dev_size)
        img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
        model = md.Model(data_dir, img_path, model_config).to(device)
        model.load_state_dict(torch.load(glob.glob(model_dir + '/*model*')[0]))
        model.eval()
        # Evaluate on the training data
        print("Training Data evaluation")
        _, t_acc = evaluate_dataset(data_dir, train_data, model, step_size)
        # Evaluate on the dev set
        print("Development Data evaluation")
        _, d_acc = evaluate_dataset(data_dir, dev_data, model, step_size)
        print('Accuracy:\nTrain Data {}%, {} total samples\nDev Data {}%, {} total samples'.format(
            round(100*t_acc,2), len(train_data), round(100*d_acc,2), len(dev_data)))
    return

def main(model_dir, data_dir, epochs=10, batch_size=10, learning_rate = 0.01, dropout=0.25, train_size=1000000, dev_size=1000000,
          word_linears = 2, word_activations = 'tanh', out_linears = 2, out_activations = 'relu', seed = 22, early_stop = 5, model='model'):
    train_data, dev_data = pre.preprocessData(data_dir, train_size, dev_size)
    configureWandB(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, dropout=dropout, train_size=len(train_data),
                   dev_size=len(dev_data), word_linears=word_linears, word_activations=word_activations, out_linears=out_linears,
                   out_activations=out_activations, seed=seed, early_stop=early_stop, model=model)
    train(model_dir=model_dir, data_dir=data_dir, train_data=train_data, dev_data=dev_data)
    return

if __name__ == "__main__":
    # sets up a command-line interface for "train" and "predict"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=main)
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--dropout", type=float, default=0.25)
    train_parser.add_argument("--train_size", type=int, default=1000000)
    train_parser.add_argument("--dev_size", type=int, default=1000000)
    train_parser.add_argument("--word_linears", type=int, default=1)
    train_parser.add_argument("--word_activations", type=str, default='tanh')
    train_parser.add_argument("--out_linears", type=int, default=2)
    train_parser.add_argument("--out_activations", type=str, default='relu')
    train_parser.add_argument("--seed", type=int, default=22)
    train_parser.add_argument("--early_stop", type=int, default=100)
    train_parser.add_argument("--model", type=str, default='model')
    
    predict_parser = subparsers.add_parser("evaluate")
    predict_parser.set_defaults(func=evaluate)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("data_dir")
    predict_parser.add_argument("--test_dir", type=str, default=None)
    predict_parser.add_argument("--step_size", type=int, default=2)
    predict_parser.add_argument("--train_size", type=int, default=1000000)
    predict_parser.add_argument("--dev_size", type=int, default=1000000)
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)