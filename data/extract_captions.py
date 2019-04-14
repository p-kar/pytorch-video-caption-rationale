import os
import re
import csv
import pdb
import json
import random

def extract_captions(opts):

    caption_file = os.path.join(opts.data_dir, 'multilingual_corpus.csv')
    video_clips_dir = os.path.join(opts.data_dir, 'clips/')
    video_clips = set([f for f in os.listdir(video_clips_dir) if f.endswith('.avi')])
    language = 'English'

    with open(caption_file) as fp:
        reader = csv.DictReader(fp)
        captions = [row for row in reader if row['Language'] == language]

    valid_captions = [c for c in captions if '{}_{}_{}.avi'.format(c['VideoID'], c['Start'], c['End']) in video_clips]
    print('Found {} captions in {}'.format(len(captions), caption_file))
    print('Dropping {} captions because of missing video files'.format(len(captions) - len(valid_captions)))

    caption_dict = {}
    for caption in valid_captions:
        video_key = '{}_{}_{}'.format(caption['VideoID'], caption['Start'], caption['End'])
        if video_key in caption_dict:
            caption_dict[video_key]['captions'].append({'desc': caption['Description'], 'source': caption['Source']})
        else:
            caption_dict[video_key] = {}
            caption_dict[video_key]['video_key'] = video_key
            caption_dict[video_key]['video_id'] = caption['VideoID']
            caption_dict[video_key]['start'] = caption['Start']
            caption_dict[video_key]['end'] = caption['End']
            caption_dict[video_key]['lang'] = caption['Language']
            caption_dict[video_key]['captions'] = [{'desc': caption['Description'], 'source': caption['Source']}]
            caption_dict[video_key]['file_path'] = os.path.join('clips/', video_key + '.avi')

    video_caption_list = list(caption_dict.values())
    random.shuffle(video_caption_list)
    train_perc = 0.85
    train_idx = int(len(video_caption_list) * train_perc)
    with open(os.path.join(opts.data_dir, 'train_captions.json'), 'w') as fp:
        json.dump(video_caption_list[:train_idx], fp)
    with open(os.path.join(opts.data_dir, 'val_captions.json'), 'w') as fp:
        json.dump(video_caption_list[train_idx:], fp)

