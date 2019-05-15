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
from torchvision import models, transforms

use_cuda = torch.cuda.is_available()

def load_frame(frame, img_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])
    return transform(frame)

def preprocess_frame(image, target_height, target_width):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def extract_frames(video_file, img_size):
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = preprocess_frame(frame, img_size, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * 255
        frame = Image.fromarray(np.uint8(frame))
        frame_list.append(frame)

    return frame_list


def extract_video_feats(opts):
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
    model.eval()

    for video in tqdm(video_clips):
        video_path = os.path.join(video_clips_dir, video)
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

        frame_list = extract_frames(video_path, opts.img_size)

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

