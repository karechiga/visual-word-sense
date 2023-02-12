"""

"""
import os
import numpy as np
import glob
import json
import torch


def createPretrainedNLPFile(data_dir):
    vocab = dict()
    with open(data_dir + '/embeddings/glove.6B.50d.txt','rt', encoding="utf8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab[i_word] = i_embeddings
    with open(data_dir + "/embeddings/vocab.json", "w") as outfile:
        outfile.write(json.dumps(vocab, indent=4))
    return

def getGloVeVocab(data_dir):
    # Create a json dictionary of pretrained word embeddings (if it doesn't already exist)
    if not os.path.exists(data_dir + "/embeddings/vocab.json"):
        createPretrainedNLPFile(data_dir)
    vocab = json.load(open(data_dir + "/embeddings/vocab.json",'r'))
    vocab_npa = np.vstack(list(vocab.keys()))
    #insert '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<unk>')
    return vocab_npa

def getGloVeEmbeddings(data_dir):
    # Create a json dictionary of pretrained word embeddings (if it doesn't already exist)
    if not os.path.exists(data_dir + "/embeddings/vocab.json"):
        createPretrainedNLPFile(data_dir)
    vocab = json.load(open(data_dir + "/embeddings/vocab.json",'r'))
    embs_npa = np.vstack(list(vocab.values()))
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    #insert embeddings for unk tokens at top of embs_npa.
    embs_npa = np.vstack((unk_emb_npa,embs_npa))
    return embs_npa

def getTokenizedVocab(data_dir):
    # Returns a dict of all the tokens in the vocab
    if not os.path.exists(data_dir + "/embeddings/tokens.json"):
        vocab = getGloVeVocab(data_dir)
        tokens = dict()
        for i, v in enumerate(vocab):
            tokens[v] = i
        with open(data_dir + "/embeddings/tokens.json", "w") as outfile:
            outfile.write(json.dumps(tokens, indent=4))
    else:
        tokens = json.load(open(data_dir + "/embeddings/tokens.json",'r'))
    return tokens

def wordEmbeddingLayer(data_dir):
    emb = getGloVeEmbeddings(data_dir)
    layer =  torch.nn.Embedding.from_pretrained(torch.from_numpy(emb).float())
    return layer

def getImageEmbeddings(data_dir):
    # Create a dictionary of images
    imgs = []
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    for img in os.listdir(img_path):
        imgs.append(img)
        # imgs[img] = cv.imread(img_path + img)