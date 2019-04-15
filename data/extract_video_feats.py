import os
import re
import csv
import pdb
import glob
import json
import random
import shutil
import subprocess

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import models, transforms

use_cuda = torch.cuda.is_available()

def load_frame(fname, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(fname)
    return transform(img)

def extract_frames(video_file, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video_file,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_video_feats(opts):
    """Extract ImageNet features from video clips"""
    corpus_base_dir = os.path.join(opts.data_dir, opts.corpus)
    video_clips_dir = os.path.join(corpus_base_dir, 'clips/')
    if opts.corpus == 'msvd':
        file_ext = '.avi'
    elif opts.corpus == 'msr-vtt':
        file_ext = '.mp4'
    else:
        raise NotImplementedError('unknown corpus')
    video_clips = [f for f in os.listdir(video_clips_dir) if f.endswith(file_ext)]
    feats_dir = os.path.join(corpus_base_dir, 'feats/')
    if os.path.exists(feats_dir):
        shutil.rmtree(feats_dir)
    os.makedirs(feats_dir)
    frames_dir = os.path.join(corpus_base_dir, 'frames/')
    
    if opts.vision_arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential()
    elif opts.vision_arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Sequential()
    else:
        raise NotImplementedError('unknown vision architecture')

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    for video in tqdm(video_clips):
        video_path = os.path.join(video_clips_dir, video)
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

        extract_frames(video_path, frames_dir)

        frame_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        if len(frame_list) > opts.num_frames:
            frame_indices = np.linspace(0, len(frame_list), num=opts.num_frames, endpoint=False).astype(int)
        else:
            frame_indices = np.arange(0, len(frame_list))
        frames = torch.stack([load_frame(frame_list[idx], opts.img_size) for idx in frame_indices])
        frames = frames.to(device)

        with torch.no_grad():
            feats = model(frames).data.cpu().numpy()

        feat_save_path = os.path.join(feats_dir, video_base_name + '.npy')
        np.save(feat_save_path, feats)

