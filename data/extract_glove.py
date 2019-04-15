import os
import re
import csv
import pdb
import sys
import json
import random
import shutil

from nltk import word_tokenize
sys.path.append('..')
from utils import load_glove_file

def extract_glove(opts):

    corpus_base = os.path.join(opts.data_dir, opts.corpus)
    captions = []
    with open(os.path.join(corpus_base, 'train_captions.json'), 'r') as fp:
        content = json.load(fp)
        for c in content:
            captions.extend(c['captions'])

    vocab = set([])
    for caption in captions:
        words = word_tokenize(caption['desc'])
        words = [word.lower() for word in words]
        vocab.update(words)

    print('Found {} words in caption vocabulary'.format(len(vocab)))

    glove_dir = os.path.join(opts.data_dir, 'glove/')
    trunc_glove_dir = os.path.join(corpus_base, 'glove/')
    if os.path.isdir(trunc_glove_dir):
        shutil.rmtree(trunc_glove_dir)
    os.makedirs(trunc_glove_dir)
    glove_files = sorted([f for f in os.listdir(glove_dir) if f.endswith('.txt')])
    vocab = list(vocab)

    for glove_file in glove_files:
        print('Processing {}... '.format(glove_file), end='', flush=True)
        word_to_index, index_to_word, word_vectors = load_glove_file(os.path.join(glove_dir, glove_file))
        words = []
        indexes = []
        for word in vocab:
            if word not in word_to_index:
                continue
            else:
                words.append(word)
                indexes.append(word_to_index[word])

        file_path = os.path.join(trunc_glove_dir, glove_file)
        with open(file_path, 'w') as fp:
            for word, idx in zip(words, indexes):
                fp.write(word + ' ' + ' '.join([str(x) for x in word_vectors[idx]]) + '\n')
        print('done')

