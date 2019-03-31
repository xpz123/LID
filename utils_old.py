import re
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.datasets import TranslationDataset as Trans
import nltk
import random as rd
import pdb


def load_dataset(batch_size):
    #tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    def tokenize_enword(text):
        return tokenizer.tokenize(text)
        #return [tok for tok in text.strip().split()]

    def tokenzie_cncha(text):
        return [tok for tok in text.strip()]

    def tokenzie_cnword(text):
        return [tok for tok in text.strip().split()]


    ZH_CHA = Field(tokenize=tokenzie_cncha, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    ZH_WORD = Field(tokenize=tokenzie_cnword, include_lengths=True,
            init_token='<sos>', eos_token='<eos>')

    EN_WORD = Field(tokenize=tokenize_enword, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')


    train, val, test = Trans.splits(path='data/', exts=('.zh', '.en'),
         fields=(ZH_WORD, EN_WORD), train='train')

    ZH_WORD.build_vocab(train.src, min_freq=30)
    EN_WORD.build_vocab(train.trg, min_freq=30)


    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, ZH_WORD, EN_WORD
