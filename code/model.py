
import torchvision.models as models
import torch
import embeddings as emb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(torch.nn.Module):
    def __init__(self, data_dir, img_path, config):
        super(Model, self).__init__()
        self.img_path = img_path
        self.w_embeddings = emb.wordEmbeddingLayer(data_dir)
        i_weights = models.ResNet50_Weights.IMAGENET1K_V2
        res50 = models.resnet50(weights=i_weights)
        layers = list(res50._modules.keys())
        # Want to remove the final linear layer of ResNet50
        self.i_pretrained = torch.nn.Sequential(*[res50._modules[x] for x in layers[:-1]])
        self.i_preprocess = i_weights.transforms()
        res50 = None
        i_weights = None
        layers = None
        w_activation = torch.nn.ReLU() if config['word_activations'] == 'relu' else torch.nn.Tanh()
        if config['word_linears'] == 1:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 2048, device=device), w_activation)
        elif config['word_linears'] == 2:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 512, device=device), w_activation,
                                                    torch.nn.Linear(512, 2048, device=device), w_activation)
        elif config['word_linears'] == 3:
            self.w_sequential = torch.nn.Sequential(torch.nn.Linear(300, 512, device=device), w_activation,
                                                    torch.nn.Linear(512, 1024, device=device), w_activation,
                                                    torch.nn.Linear(1024, 2048, device=device), w_activation)
        o_activation = torch.nn.ReLU() if config['out_activations'] == 'relu' else torch.nn.Tanh()
        drop = torch.nn.Dropout(p=config['dropout'])
        if config['out_linears'] == 1:
            self.o_sequential = torch.nn.Sequential(drop, torch.nn.Linear(6144, 1, device=device))
        elif config['out_linears'] == 2:
            self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                    o_activation, torch.nn.Linear(3072, 1, device=device))
        elif config['out_linears'] == 3:
            self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 1, device=device))
        elif config['out_linears'] == 4:
            self.o_sequential = torch.nn.Sequential(torch.nn.Linear(6144, 3072, device=device), drop,
                                                    o_activation, torch.nn.Linear(3072, 1024, device=device),
                                                    o_activation, torch.nn.Linear(1024, 512, device=device),
                                                    drop, o_activation, torch.nn.Linear(512, 1, device=device))
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
        num_img_samp = int(i_seqs.shape[0]/w_seq.shape[0])
        # for each sample in the batch, then for each image in each sample, concatenate the word tensors to the image tensors
        for i in range(w_seq.shape[0]):
            samps = []
            for j in range(i*num_img_samp, i*num_img_samp + num_img_samp):
                x = torch.cat((i_seqs[j].unsqueeze(0), w_seq[i])).to(device)
                samps.append(x.flatten())
            out_tensor.append(torch.stack(samps).to(device))
        out = self.o_sequential(torch.stack(out_tensor).to(device)).squeeze(2)
        # print("forward output: {} GB".format(psutil.Process(os.getpid()).memory_info().rss/1000000000))
        return out