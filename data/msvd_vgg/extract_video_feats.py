import os
import re
import csv
import pdb
import glob
import json
import random
import shutil
import subprocess

import numpy as np
from tqdm import tqdm

def extract_video_feats(opts):
    """Extract ImageNet features from video clips"""
    corpus_base_dir = os.path.join(opts.data_dir, opts.corpus)
    video_feat_files = glob.glob(os.path.join(corpus_base_dir, 'yt_allframes_vgg_fc7_*'))
    feats_dir = os.path.join(corpus_base_dir, 'feats/')
    if os.path.exists(feats_dir):
        shutil.rmtree(feats_dir)
    os.makedirs(feats_dir)
    vid_name_regex = r"(vid\d+)_frame_(\d+)"

    for video_feat_file in video_feat_files:

        with open(video_feat_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [row for row in reader]

        video_feats = {}
        for row in rows:
            assert len(row) == 4097
            match_obj = re.match(vid_name_regex, row[0])
            video_base_name = match_obj[1]
            video_frame_num = int(match_obj[2])
            feat = np.array([float(x) for x in row[1:]])

            if video_base_name in video_feats:
                video_feats[video_base_name].append((video_frame_num, feat))
            else:
                video_feats[video_base_name] = [(video_frame_num, feat)]

        for video_base_name in video_feats.keys():
            frames = sorted(video_feats[video_base_name], key=lambda x: x[0])
            frames = np.array([f[1] for f in frames])

            if len(frames) > opts.num_frames:
                frame_indices = np.linspace(0, len(frames), num=opts.num_frames, endpoint=False).astype(int)
            else:
                frame_indices = np.arange(0, len(frames))
            feats = frames[frame_indices]
            feat_save_path = os.path.join(feats_dir, video_base_name + '.npy')
            np.save(feat_save_path, feats)

