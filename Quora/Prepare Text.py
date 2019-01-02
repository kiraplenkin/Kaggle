#%%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

#%%
train = pd.read_csv('/Users/kirr/Pyth/Notebooks/Quora/data/train.csv')
test = pd.read_csv('/Users/kirr/Pyth/Notebooks/Quora/data/test.csv')

#%%
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

#%%
def build_vocab(sentences, verbose=True):
    # param sentences: list of list of words
    # return: dictionary of words and their count
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

#%%
sentences = train['question_text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})

#%%
