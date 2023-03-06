"""

"""
import os
import numpy as np
import glob
import json
import torch

def readGloVeFile(data_dir):
    vocab = dict()
    try:
        with open(data_dir + '/embeddings/glove.42B.300d.txt','rt', encoding="utf8") as fi:
            full_content = fi.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab[i_word] = i_embeddings
    except:
        try:
            vocab = json.load(open(data_dir + "/embeddings/vocab.json",'r'))
        except:
            raise Exception('Could not load embedding file.')
    return vocab

# def createPretrainedNLPFile(data_dir):
#     vocab = readGloVeFile(data_dir)
#     with open(data_dir + "/embeddings/vocab.json", "w") as outfile:
#         outfile.write(json.dumps(vocab, indent=4))
#         # save as torch?
#     return vocab

def getGloVeVocab(data_dir):
    vocab = readGloVeFile(data_dir)
    vocab_npa = np.vstack(list(vocab.keys()))
    #insert '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<unk>')
    return vocab_npa

def getGloVeEmbeddings(data_dir):
    vocab = readGloVeFile(data_dir)
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
    return torch.nn.Embedding.from_pretrained(
        torch.from_numpy(getGloVeEmbeddings(data_dir)).float())

def getImageEmbeddings(data_dir):
    # Create a dictionary of images
    imgs = []
    img_path = glob.glob(data_dir + '/*train*/*images*/')[0]
    for img in os.listdir(img_path):
        imgs.append(img)
        # imgs[img] = cv.imread(img_path + img)

def tokenize(words, data_dir):
    # takes in a 2D list of words and returns a 2D list of their corresponding tokens.
    tokens = getTokenizedVocab(data_dir)
    out = np.zeros(shape=(len(words), len(words[0])))
    for i in range(len(words)):
        for j, word in enumerate(words[i]):
            try:
                out[i,j] = tokens[word]
            except:
                out[i,j] = 0    # token is zero for unknown tokens
    return out

def untokenize(tokens, data_dir):
    vocab = getTokenizedVocab(data_dir)
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