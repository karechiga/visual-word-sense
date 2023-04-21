
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sentence_transformers import SentenceTransformer
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device.type}')
class Model(torch.nn.Module):
    def __init__(self, data_dir, img_path, config):
        super(Model, self).__init__()
        self.img_path = img_path
        self.data_dir = data_dir
        self.w_model = config['model']
        if config['model'] == 'glove':
            self.w_embeddings = emb.wordEmbeddingLayer(data_dir)
            o_dim = 2048*3
            w_embed_dim = 300
        else:
            self.transformer_model = SentenceTransformer('sentence-transformers/all-' + config['model'])
            o_dim = 2048*2
            w_embed_dim = self.transformer_model._modules['1'].word_embedding_dimension
        i_weights = models.ResNet50_Weights.IMAGENET1K_V2
        res50 = models.resnet50(weights=i_weights)
        layers = list(res50._modules.keys())
        # Want to remove the final linear layer of ResNet50
        i_drop = torch.nn.Dropout(p=config['i_dropout'] if 'i_dropout' in config.keys() else 0)
        self.i_pretrained = torch.nn.Sequential(*[res50._modules[x] for x in layers[:-1]], i_drop)
        self.i_preprocess = i_weights.transforms()
        res50 = None
        i_weights = None
        layers = None
        w_activation = torch.nn.ReLU() if config['word_activations'] == 'relu' else torch.nn.Tanh()
        self.weighted_sum = None
        self.dot_w = None
        if config['vector_combine'] == 'dot':
            self.dot_w = torch.nn.Parameter(torch.randn(1,2048))
            if config['model'] == 'glove':
                # Combines the 2 words
                self.weighted_sum = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        w_drop = torch.nn.Dropout(p=config['w_dropout'] if 'w_dropout' in config.keys() else 0)
        if config['word_linears'] == 1:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(w_embed_dim, 2048, device=device), w_activation, w_drop)
        elif config['word_linears'] == 2:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(w_embed_dim, 512, device=device), w_activation,
                                                    torch.nn.Linear(512, 2048, device=device), w_activation, w_drop)
        elif config['word_linears'] == 3:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(w_embed_dim, 512, device=device), w_activation,
                                                    torch.nn.Linear(512, 1024, device=device), w_activation,
                                                    torch.nn.Linear(1024, 2048, device=device), w_activation, w_drop)
        o_drop = torch.nn.Dropout(p=config['dropout'] if 'dropout' in config.keys() else 0)
        o_activation = torch.nn.ReLU() if config['out_activations'] == 'relu' else torch.nn.Tanh()
        if config['out_linears'] == 1:
            if 'dropout' in config.keys():
                self.o_sequential = torch.nn.Sequential(o_drop, torch.nn.Linear(o_dim, 1, device=device))
            else:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 1, device=device))
        elif config['out_linears'] == 2:
            if 'dropout' in config.keys():
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device), o_drop,
                                                    o_activation, torch.nn.Linear(3072, 1, device=device))
            else:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device),
                                                    o_activation, torch.nn.Linear(3072, 1, device=device))
        elif config['out_linears'] == 3:
            if 'dropout' in config.keys():
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device), o_drop,
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 1, device=device))
            else:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device),
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 1, device=device))
        elif config['out_linears'] == 4:
            if 'dropout' in config.keys():
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device), o_drop,
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 512, device=device),
                                                    o_drop, o_activation, torch.nn.Linear(512, 1, device=device))
            else:
                self.o_sequential = torch.nn.Sequential(torch.nn.Linear(o_dim, 3072, device=device),
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 512, device=device),
                                                    o_activation, torch.nn.Linear(512, 1, device=device))
    def image_model(self, samples):
        # Returns outputs of NNs with input of multiple images
        batch = []
        for n, samp in enumerate(samples):
            for i in samp:
                img = Image.open(self.img_path + i).convert('RGB')
                img = self.i_preprocess(img).unsqueeze(0).to(device)
                batch.append(img)
        # Input all the images in the batch to the resnet model
        x = self.i_pretrained(torch.stack(batch).squeeze(1).to(device)).to(device).squeeze(2).squeeze(2)
        # Outputs a BatchSize*10 x 2048 Tensor
        return x
    def word_model(self, words):
        # Returns the output of a NN with an input of two words
        if self.w_model == 'glove':
            # inputs are tokens in this case
            x = self.w_embeddings(words.to(device))
        else:
            # input is a string
            x = torch.Tensor(self.transformer_model.encode(words)).to(device)
        if self.weighted_sum is not None:
            x = self.weighted_sum(x)
        x = self.w_sequential(x)
        return x
    def forward(self, words, images):
        # Networks to be used for the images of the dataset
        i_seqs = self.image_model(images)
        # print("images pretrained: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        # Network to be used for the words of the dataset
        w_seq = self.word_model(words)
        # Combining the word and image tensors
        out_tensor = []
        num_img_samp = int(i_seqs.shape[0]/w_seq.shape[0])
        # for each sample in the batch, then for each image in each sample, concatenate the word tensors to the image tensors
        for i in range(w_seq.shape[0]):
            samps = []
            for j in range(i*num_img_samp, i*num_img_samp + num_img_samp):
                if self.dot_w is not None:
                    x = torch.dot(self.dot_w.squeeze(0) * i_seqs[j], w_seq[i].squeeze(0))
                else:
                    x = torch.cat((i_seqs[j].unsqueeze(0), w_seq[i].unsqueeze(0) if len(w_seq[i].shape) == 1 else w_seq[i])).to(device)
                    x = x.flatten()
                samps.append(x)
            out_tensor.append(torch.stack(samps).to(device))
        if self.dot_w is None:
            out = self.o_sequential(torch.stack(out_tensor).to(device)).squeeze(2)
        else:
            out = torch.stack(out_tensor).to(device)
        return out