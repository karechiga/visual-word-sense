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
import wandb
wandb.login()
import preprocess as pre
import main as mn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_config = {
    "learning_rate": 0.00058,
    "batch_size": 32,
    "epochs": 3,
    "train_size": 500,
    "dev_size": 200,
    "dropout": 0.25
    }

def main():
    model_dir = "../models_data"
    data_dir = "../data"
    wandb.init(project="visualwordsense")
    wandb.config.setdefaults(default_config)
    
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
            self.linear3 = torch.nn.Linear(6144, 3072, device=device)
            self.linear4 = torch.nn.Linear(3072, 1, device=device)
            self.relu = torch.nn.ReLU()
            self.tanh = torch.nn.Tanh()
            self.drop = torch.nn.Dropout(p=wandb.config['dropout'])
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
                    x = torch.cat((img.unsqueeze(0), w_seq[i])).to(device)
                    samps.append(x.flatten())
                out_tensor.append(torch.stack(samps).to(device))
            out = self.linear3(torch.stack(out_tensor).to(device))
            out = self.relu(out)
            out = self.drop(out)
            out = self.linear4(out).squeeze(2)
            # out = self.softmax(out).to(device)
            # print("forward output: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
            return out

    train_data, dev_data = pre.preprocessData(data_dir, wandb.config.train_size, wandb.config.dev_size)
    wandb.config.train_size = len(train_data)
    wandb.config.dev_size = len(dev_data)
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    model = Model(data_dir, img_path).to(device)
    loss_fnc = torch.nn.CrossEntropyLoss()
    # loss_fnc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    best_vloss = 1_000_000.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('{} - lr{} batchsize{} epochs{} train_size{} dev_size{}'.format(
        timestamp, wandb.config.learning_rate, wandb.config.batch_size, wandb.config.epochs, len(train_data), len(dev_data)))
    model_dir = model_dir + '/{}_lr{}_b{}_e{}_train{}'.format(
        timestamp, round(wandb.config.learning_rate,5), wandb.config.batch_size, wandb.config.epochs, len(train_data))
    print('Model dir: {}'.format(model_dir))
    os.mkdir(model_dir)
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
        print('Accuracy:\ttrain {}%\tvalidation {}%'.format(round(100*accuracy,2), round(val_acc,2)))

        # Track best performance, and save the model's state
        if val_acc > 0.70:  # Only save model if it is greater than 50% accuracy on the dev set
            print('saving model')
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
        'name': 'lr_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
            },
        'parameters': {
            'dropout': {'max': 0.9, 'min': 0.10},
            'learning_rate': {'max': 0.0002, 'min': 0.000001},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="visualwordsense")

    wandb.agent(sweep_id, project='visualwordsense',function=main, count=10)
