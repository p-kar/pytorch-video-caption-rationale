import os
import re
import csv
import pdb
import glob
import json
import random
import shutil
import subprocess

import cv2
import torch
import skimage
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from model.darknet import Darknet

use_cuda = torch.cuda.is_available()

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0)
    return img_

def extract_frames(video_file, img_size):
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
    return frame_list


def extract_bbox_feats(opts):
    """Extract ImageNet features from video clips"""
    corpus_base_dir = os.path.join(opts.data_dir, opts.corpus)
    video_clips_dir = os.path.join(corpus_base_dir, 'clips/')
    if opts.corpus == 'msvd':
        file_ext = '.avi'
    elif opts.corpus == 'msrvtt':
        file_ext = '.mp4'
    else:
        raise NotImplementedError('unknown corpus')
    video_clips = [f for f in os.listdir(video_clips_dir) if f.endswith(file_ext)]
    feats_dir = os.path.join(corpus_base_dir, 'bbox_feats/')
    if os.path.exists(feats_dir):
        shutil.rmtree(feats_dir)
    os.makedirs(feats_dir)
    
    print("Loading network.....")
    model = Darknet(os.path.join(opts.data_dir, 'yolo/', 'yolov3.cfg'))
    model.load_weights(os.path.join(opts.data_dir, 'yolo/', 'yolov3.weights'))
    model.net_info["height"] = opts.img_size
    print("Network successfully loaded")
    assert opts.img_size % 32 == 0
    assert opts.img_size > 32

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    for video in tqdm(video_clips):
        video_path = os.path.join(video_clips_dir, video)
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_list = extract_frames(video_path, opts.img_size)

        if len(frame_list) > opts.num_frames:
            frame_indices = np.linspace(0, len(frame_list), num=opts.num_frames, endpoint=False).astype(int)
        else:
            frame_indices = np.arange(0, len(frame_list))
        frames = torch.stack([prep_image(frame_list[idx], opts.img_size) for idx in frame_indices])
        frames = frames.to(device)

        with torch.no_grad():
            feats = model.get_feats(frames).data.cpu().numpy()

        feat_save_path = os.path.join(feats_dir, video_base_name + '.npy')
        np.save(feat_save_path, feats)

