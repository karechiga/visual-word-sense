import argparse
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import wandb
wandb.login()
import preprocess as pre
import main as mn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_config = {
    "learning_rate": 0.00058,
    "batch_size": 32,
    "epochs": 3,
    "train_size": 600,
    "dev_size": 200,
    'random_seed': 22,
    'early_stop': 1000,
    'early_stop_threshold': 0.01,
    'model' : 'mpnet_base_v2',
    'dropout': 0.25,
    'word_linears': 1,
    'word_activations': 'tanh',
    'out_linears': 2,
    'out_activations': 'relu',
    'sent_structure': 'word0 word1'
}

def main():
    wandb.init(project="visualwordsense")
    wandb.config.setdefaults(default_config)
    np.random.seed(wandb.config.random_seed)
    train_data, dev_data, train_img, dev_img = pre.preprocessData(data_dir, kwargs['train_size'], kwargs['dev_size'])
    mn.train(model_dir=model_dir, data_dir=data_dir, train_data=train_data, dev_data=dev_data, 
             train_img=train_img, dev_img=dev_img, save=False)
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--early_stop", type=int, default=1000)
    parser.add_argument("--es_threshold", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train_size", type=int, default=1000000)
    parser.add_argument("--dev_size", type=int, default=1000000)
    parser.add_argument("--model", type=str, default='mpnet_base_v2')
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--sent_structure", type=str, default='word0 word1')

    # parser.add_argument('--bn_not_track', action='store_false')

    args = parser.parse_args()
    kwargs = vars(args)
    default_config['epochs'] = kwargs['epochs']
    default_config['batch_size'] = kwargs['batch_size']
    default_config['early_stop'] = kwargs['early_stop']
    default_config['early_stop_threshold'] = kwargs['es_threshold']
    default_config['learning_rate'] = kwargs['learning_rate']
    default_config['train_size'] = kwargs['train_size']
    default_config['dev_size'] = kwargs['dev_size']
    default_config['dropout'] = kwargs['dropout']
    default_config['model'] = kwargs['model']
    default_config['sent_structure'] = kwargs['sent_structure']

    if kwargs['sweep_id'] is not None:
        sweep_id = kwargs['sweep_id']
    else:
        sweep_config = {
        'method': 'random',
        'name': 'split_images',
        'metric': {
            'goal': 'maximize',
            'name': 'best_val_acc'
            },
        'parameters': {
            'dropout': {'max': 0.65, 'min': 0.1},
            'learning_rate': {'max': 0.00065, 'min': 0.00045},
            'random_seed': {'max' : 15, 'min': 0},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="visualwordsense")

    model_dir = "../models_data"
    data_dir = "../data"
    wandb.agent(sweep_id, project='visualwordsense',function=main, count=10)
