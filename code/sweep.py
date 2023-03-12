import argparse
import glob
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
import os
import numpy as np
import wandb
wandb.login()
import preprocess as pre
import main as mn
import model as md

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_config = {
    "learning_rate": 0.00058,
    "batch_size": 32,
    "epochs": 3,
    "train_size": 500,
    "dev_size": 200,
    "dropout": 0.25,
    'word_linears': 2,
    'word_activations': 'tanh',
    'out_linears': 2,
    'out_activations': 'relu',
    'random_seed': 22
    }

def main():
    model_dir = "../models_data"
    data_dir = "../data"
    wandb.init(project="visualwordsense")
    wandb.config.setdefaults(default_config)

    train_data, dev_data = pre.preprocessData(data_dir, wandb.config.train_size, wandb.config.dev_size)
    np.random.seed(wandb.config.random_seed)
    wandb.config.update({'train_size':len(train_data), 'dev_size':len(dev_data)}, allow_val_change=True)
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = md.Model(data_dir, img_path).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    # loss_fnc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    best_vloss = 1_000_000.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('{} - lr{} batchsize{} epochs{} train_size{} dev_size{}'.format(
        timestamp, wandb.config.learning_rate, wandb.config.batch_size, wandb.config.epochs, len(train_data), len(dev_data)))
    for e in range(wandb.config.epochs):
        batches = pre.getBatches(train_data, wandb.config.batch_size)
        vbatches = pre.getBatches(dev_data, wandb.config.batch_size) # validation data split into batches
        print('batches created')
        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = None
        accuracy = 0
        avg_loss, accuracy = mn.train_epoch(e, batches, model, loss_fnc, optimizer)
        wandb.log({"train_acc": accuracy, "train_loss": avg_loss})
        model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e+1)
        val_loss = None
        val_acc = 0
        val_loss, val_acc = mn.eval_epoch(vbatches, model, loss_fnc)
        wandb.log({"val_acc": val_acc, "val_loss": val_loss})
        print('Loss:\t\ttrain {}\tvalidation {}'.format(round(avg_loss,2), round(val_loss,2)))
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(round(100*accuracy,2), round(100*val_acc,2)))

        # Track best performance, and save the model's state
        if val_acc > 0.70:  # Only save model if it is greater than 50% accuracy on the dev set
            print('saving model')
            model_dir = model_dir + '/{}_lr{}_b{}_e{}_train{}'.format(
                timestamp, round(wandb.config.learning_rate,5), wandb.config.batch_size, wandb.config.epochs, len(train_data))
            os.mkdir(model_dir)
            print('Model dir: {}'.format(model_dir))
            model_path = model_dir + '/' + 'model_{}_{}'.format(timestamp, e)
            torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train_size", type=int, default=1000000)
    parser.add_argument("--dev_size", type=int, default=1000000)
    args = parser.parse_args()
    kwargs = vars(args)
    default_config['epochs'] = kwargs['epochs']
    default_config['batch_size'] = kwargs['batch_size']
    default_config['learning_rate'] = kwargs['learning_rate']
    default_config['train_size'] = kwargs['train_size']
    default_config['dev_size'] = kwargs['dev_size']
    default_config['dropout'] = kwargs['dropout']
    
    if kwargs['sweep_id'] is not None:
        sweep_id = kwargs['sweep_id']
    else:
        sweep_config = {
        'method': 'random',
        'name': 'new_model1',
        'metric': {
            'goal': 'maximize',
            'name': 'val_acc'
            },
        'parameters': {
            'dropout': {'max': 0.9, 'min': 0.10},
            'learning_rate': {'max': 0.001, 'min': 0.0003},
            'word_linears': {'values' : [1, 2, 3]},
            'out_linears': {'values' : [1, 2, 3, 4]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="visualwordsense")

    wandb.agent(sweep_id, project='visualwordsense',function=main, count=10)
