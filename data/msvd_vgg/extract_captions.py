import os
import re
import csv
import pdb
import json
import random

def extract_captions_file(corpus_dir, split):
    caption_file = os.path.join(corpus_dir, 'sents_{}_lc_nopunc.txt'.format(split))

    with open(caption_file, 'r') as fp:
        content = fp.readlines()

    caption_dict = {}
    for line in content:
        line = line.strip().split('\t')
        video_key = line[0]
        caption = ' '.join(line[1:])

        if video_key in caption_dict:
            caption_dict[video_key]['captions'].append({'desc': caption})
        else:
            caption_dict[video_key] = {}
            caption_dict[video_key]['video_key'] = video_key
            caption_dict[video_key]['captions'] = [{'desc': caption}]

    video_caption_list = list(caption_dict.values())
    with open(os.path.join(corpus_dir, '{}_captions.json'.format(split)), 'w') as fp:
        json.dump(video_caption_list, fp)


def extract_captions(opts):

    corpus_dir = os.path.join(opts.data_dir, opts.corpus)
    splits = ['train', 'val', 'test']

    for split in splits:
        extract_captions_file(corpus_dir, split)

