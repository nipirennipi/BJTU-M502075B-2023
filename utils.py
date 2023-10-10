import pandas as pd
import random
import re
import numpy as np
import torch

from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm

GREEN = "\033[32m"
RESET = "\033[0m"


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = text.lower()
    tokenized_text = word_tokenize(text)
    return tokenized_text


def read_news(file_path, filter_num):
    column_names = [
        'nid', 'cate', 'subcate', 'title', 'abstract', 'url'
    ]
    raw_data = pd.read_csv(
        file_path, 
        sep='\t', 
        header=None, 
        names=column_names,
    )
    word_count = Counter()
    news_dict = {}

    for idx, row in tqdm(raw_data.iterrows()):
        row['title'] = tokenizer(row['title'])
        word_count.update(row['title'])
        news_dict[row['nid']] = {'title': row['title']}
    
    # Build a vocabulary of news titles. (filter low frequency words)
    vocab = [
        word for word, cnt in word_count.items() if cnt >= filter_num
    ]
    vocab = {word: idx + 1 for idx, word in enumerate(vocab)}
    return news_dict, vocab


def load_word_vectors(vectors_path, vocab):
    # Pre-trained word vectors, and unknown words excluded.
    word_vectors = {}
    
    with open(vectors_path, 'r') as f:
        for line in tqdm(f):
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                word_vectors[vals[0]] = [float(x) for x in vals[1:]]

    return word_vectors


def green_print(values):
    print(GREEN + values + RESET)
