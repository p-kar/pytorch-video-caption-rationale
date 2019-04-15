import os
import re
import csv
import pdb
import json
import random

def extract_captions(opts):

    corpus_dir = os.path.join(opts.data_dir, opts.corpus)
    caption_file = os.path.join(corpus_dir, 'videodatainfo_2017_ustc.json')
    video_clips_dir = os.path.join(corpus_dir, 'clips/')
    video_clips = set([f for f in os.listdir(video_clips_dir) if f.endswith('.mp4')])

    with open(caption_file) as fp:
        content = json.load(fp)

    video_list = [c for c in content['videos'] if c['video_id'] + '.mp4' in video_clips]
    random.shuffle(video_list)
    print('Found {} videos in {}'.format(len(content['videos']), caption_file))
    print('Dropping {} videos because of missing video files'.format(len(content['videos']) - len(video_list)))
    train_perc = 0.80
    train_idx = int(len(video_list) * train_perc)
    train_video_list = set([v['video_id'] for v in video_list[:train_idx]])
    val_video_list = set([v['video_id'] for v in video_list[train_idx:]])

    caption_dict = {}
    for caption in content['sentences']:
        video_id = caption['video_id']
        if video_id not in caption_dict:
            split = 'train' if video_id in train_video_list else 'val'
            caption_dict[video_id] = {'video_id': video_id, 'split': split, 'captions': []}
        caption_dict[video_id]['captions'].append({'desc': caption['caption'], 'sen_id': caption['sen_id']})

    video_caption_list = list(caption_dict.values())
    with open(os.path.join(corpus_dir, 'train_captions.json'), 'w') as fp:
        json.dump([v for v in video_caption_list if v['split'] == 'train'], fp)
    with open(os.path.join(corpus_dir, 'val_captions.json'), 'w') as fp:
        json.dump([v for v in video_caption_list if v['split'] == 'val'], fp)

