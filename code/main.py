"""
    main.py
    Trains a model to match a word with a given image out of a set of images.
    Evaluates performance of a given model on a given dataset of words and images.
"""

import argparse
import numpy as np
import glob
import torch
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

def init_model(data_dir, img_path, model_config):
    if wandb.config['model'] == 'MiniLM_L6_v2':
        import model_MiniLM_L6_v2 as md
        print('Using model_MiniLM_L6_v2.py for the model.')
    elif wandb.config['model'] == 'mpnet_base_v2':
        import model_mpnet_base_v2 as md
        print('Using model_mpnet_base_v2.py for the model.')
    else:
        import model_MiniLM_L6_v2 as md
        print('Using model_MiniLM_L6_v2.py for the model.')
    return md.Model(data_dir, img_path, model_config).to(device)

def configureWandB(learning_rate, batch_size, epochs, dropout, train_size, dev_size,
                   word_linears, word_activations, out_linears, out_activations, seed,
                   early_stop, es_threshold, model):
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
            'early_stop_threshold': es_threshold,
            'model': model,
            'word_linears': word_linears,
            'word_activations': word_activations,
            'out_linears': out_linears,
            "dropout": dropout,
            'out_activations': out_activations,
        }
    )

def run_batch(batch, model, training = False, loss_fnc = None, optimizer = None, sample_num = 0, out = False):
    model.train(training)
    if training == True and optimizer is not None:
        optimizer.zero_grad()
    words, images, labels = batch
    outputs = model.forward(words, images)
    softmax = torch.nn.Softmax(dim=1)
    predictions = torch.argmax(softmax(outputs), dim=1)
    actual = torch.Tensor([np.where(images[n] == labels[n])[0][0] for n in range(len(images))]).to(device)
    if out:   # output predictions for the batch
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
    if training == True and loss_fnc is not None and optimizer is not None:
        # Compute gradients
        loss.backward()
        # Adjust learning weights
        optimizer.step()
    return correct, total, l

def run_epoch(e, batches, model, training = False, loss_fnc = None, optimizer = None, out = False):
    running_loss = 0
    correct = 0
    total = 0
    string = 'TRAINING: ' if training == True else 'EVALUATION: '
    for i, batch in enumerate(batches):
        print(string + 'EPOCH {}: Batch {} out of {}'.format(e + 1, i+1, len(batches)))
        c, t, l = run_batch(batch, model, training=training, loss_fnc=loss_fnc, optimizer=optimizer, sample_num=total,
                            out = True if i == 0 else out)  # Output the first batch of every epoch no matter what
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

def train(model_dir, data_dir, train_data, dev_data, save):
    """
    Train the model on the given training dataset located at "data_dir".
    """
    model_config = {
        'model': wandb.config.model,
        'word_linears': wandb.config.word_linears,
        'word_activations': wandb.config.word_activations,
        'out_linears': wandb.config.out_linears,
        "dropout": wandb.config.dropout,
        'out_activations': wandb.config.out_activations,
    }
    np.random.seed(wandb.config.random_seed)
    threshold = 0
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = init_model(data_dir, img_path, model_config)

    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    best_v_acc = 0.0; best_num_epochs = 0; count = 0
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('{} - lr{} batchsize{} epochs{} train_size{} dev_size{}'.format(
        timestamp, wandb.config.learning_rate, wandb.config.batch_size, wandb.config.epochs, wandb.config.train_size, wandb.config.dev_size))
    for e in range(wandb.config.epochs):
        batches = pre.getBatches(train_data, wandb.config.batch_size)
        vbatches = pre.getBatches(dev_data, wandb.config.batch_size) # validation data split into batches
        avg_loss = None; accuracy = 0
        avg_loss, accuracy = run_epoch(e=e, batches=batches, model=model, training=True, loss_fnc=loss_fnc, optimizer=optimizer)
        val_loss = None; val_acc = 0
        val_loss, val_acc = run_epoch(e=e, batches=vbatches, model=model, loss_fnc=loss_fnc)
        wandb.log({"epoch": e, "train_acc": accuracy, "train_loss": avg_loss, "val_acc": val_acc, "val_loss": val_loss})
        print('Loss:\t\ttrain {}\tvalidation {}'.format(round(avg_loss,2), round(val_loss,2)))
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(round(100*accuracy,2), round(100*val_acc,2)))

        # save the best model into memory
        if val_acc > best_v_acc:
            best_v_acc = val_acc
            best_num_epochs = e
            best_model = None
            best_model = copy.deepcopy(model.to('cpu').state_dict())
            model.to(device)
        # early stopping if accuracy doesn't go above the threshold
        if val_acc >= threshold:
            count = 0
            threshold = val_acc + wandb.config.early_stop_threshold
        else:
            count += 1
            if count >= wandb.config.early_stop:
                print("Early stopping activated after {} consecutive epochs below the threshold ({}).".format(count, round(threshold*100,2)))
                break
    wandb.log({"best_val_acc": best_v_acc, "best_num_epochs": best_num_epochs})
    if save:
        print('saving model')
        model_dir = model_dir + '/{}_lr{}_b{}_e{}_train{}'.format(
        timestamp, round(wandb.config.learning_rate, 7), wandb.config.batch_size, best_num_epochs, wandb.config.train_size)
        os.mkdir(model_dir)
        print('Model dir: {}'.format(model_dir))
        model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, best_num_epochs)
        config_path = model_dir + '/' + 'config_{}_{}.json'.format(timestamp, best_num_epochs)
        torch.save(best_model, model_path)
        with open(config_path, "w") as outfile:
            outfile.write(json.dumps(model_config, indent=4))
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
    batches = pre.getBatches(data, step_size, rand=False)
    loss, acc = run_epoch(0, batches, model, training = False, loss_fnc = None, optimizer = None, out=True)
    return loss, acc

def evaluate(model_dir, data_dir, test_dir, step_size, train_size, dev_size, remove_unk_tokens=False):
    model_config = json.load(open(glob.glob(model_dir + '/*config*')[0],'r'))
    if test_dir is not None:
        img_path = glob.glob(test_dir + '/*images*/')[0]
        X = pre.readXData(glob.glob(test_dir + '/*data*')[0])
        y = pre.readYData(glob.glob(test_dir + '/*gold*')[0])
    else:
        train_data, dev_data = pre.preprocessData(data_dir, train_size, dev_size)
        img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
        model = init_model(data_dir, img_path, model_config)
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
          word_linears = 2, word_activations = 'tanh', out_linears = 2, out_activations = 'relu', seed = 22, early_stop = 5, 
          es_threshold = 0.01, model='model', save=False):
    train_data, dev_data = pre.preprocessData(data_dir, train_size, dev_size)
    configureWandB(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, dropout=dropout, train_size=len(train_data),
                   dev_size=len(dev_data), word_linears=word_linears, word_activations=word_activations, out_linears=out_linears,
                   out_activations=out_activations, seed=seed, early_stop=early_stop, es_threshold=es_threshold, model=model)
    train(model_dir=model_dir, data_dir=data_dir, train_data=train_data, dev_data=dev_data, save=save)
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
    train_parser.add_argument("--es_threshold", type=float, default=0.01)
    train_parser.add_argument("--model", type=str, default='MiniLM_L6_v2')
    train_parser.add_argument('--save', action='store_true')
    
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