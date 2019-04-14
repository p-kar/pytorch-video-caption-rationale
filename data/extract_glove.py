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

    caption_file = os.path.join(opts.data_dir, 'multilingual_corpus.csv')
    video_clips_dir = os.path.join(opts.data_dir, 'clips/')
    video_clips = set([f for f in os.listdir(video_clips_dir) if f.endswith('.avi')])
    language = 'English'
    vocab = set([])

    with open(caption_file) as fp:
        reader = csv.DictReader(fp)
        captions = [row for row in reader if row['Language'] == language]

    valid_captions = [c for c in captions if '{}_{}_{}.avi'.format(c['VideoID'], c['Start'], c['End']) in video_clips]

    for caption in valid_captions:
        words = word_tokenize(caption['Description'])
        words = [word.lower() for word in words]
        vocab.update(words)

    print('Found {} words in caption vocabulary'.format(len(vocab)))

    glove_dir = os.path.join(opts.data_dir, 'glove/')
    trunc_glove_dir = os.path.join(glove_dir, 'trunc/')
    if os.path.isdir(trunc_glove_dir):
        shutil.rmtree(trunc_glove_dir)
    os.makedirs(trunc_glove_dir)
    glove_files = sorted([f for f in os.listdir(glove_dir) if f.endswith('.txt')])
    vocab = list(vocab)

    for glove_file in glove_files:
        print('Processing {}...'.format(glove_file), end='', flush=True)
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

