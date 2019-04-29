import os
import re
import csv
import pdb
import glob
import json
import random
import shutil
import argparse
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

from utils import *
from model.SpatialNet import SpatialNet
from model.darknet import Darknet

use_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not use_cuda else 'cuda')

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
    Prepare image for inputting to the YOLOv3 neural network.
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0)
    return img_, img

def extract_frames(video_file, img_size, num_frames):
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)

    if len(frame_list) > num_frames:
        frame_indices = np.linspace(0, len(frame_list), num=num_frames, endpoint=False).astype(int)
    else:
        frame_indices = np.arange(0, len(frame_list))
    frame_list = [frame_list[idx] for idx in frame_indices]

    return frame_list

def load_spatial_net_model(args):
    print('Loading SpatialNet...')
    root_dir = os.path.join(args.data_dir, args.corpus)
    model_info = torch.load(args.spatial_net_file, map_location='cpu')
    opts = model_info['opts']
    args.img_size = opts.img_size
    args.num_frames = opts.num_frames
    glove_loader = GloveLoader(os.path.join(root_dir, 'glove/', opts.glove_emb_file))
    model = SpatialNet(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len, opts.arch)
    model.load_state_dict(model_info['state_dict'])
    model = model.to(device)
    model.eval()
    print("Network successfully loaded")

    return model, glove_loader

def load_yolov3_model(args):
    print("Loading YOLOv3.....")
    model = Darknet(os.path.join(args.data_dir, 'yolo/', 'yolov3.cfg'))
    model.load_weights(os.path.join(args.data_dir, 'yolo/', 'yolov3.weights'))
    model.net_info["height"] = args.img_size
    model = model.to(device)
    model.eval()
    print("Network successfully loaded")
    assert args.img_size % 32 == 0
    assert args.img_size > 32

    return model

parser = argparse.ArgumentParser(description='Evaluate spatial attention')
parser.add_argument('--data_dir', default='./data', type=str, help='root directory of the dataset')
parser.add_argument('--corpus', default='msvd', type=str, help='video captioning corpus to use')
parser.add_argument('--spatial_net_file', default='./trained_models/best_spatial.net', type=str, help='Trained SpatialNet model')
parser.add_argument('--vid_file', default='Qp-k0H93iJE_35_39.avi', type=str, help='Video file to visualize')

args = parser.parse_args()
spatial_model, glove_loader = load_spatial_net_model(args)
yolo_model = load_yolov3_model(args)

vid_path = os.path.join(args.data_dir, args.corpus, 'clips/', args.vid_file)
frame_list = extract_frames(vid_path, args.img_size, args.num_frames)
frame_tensor = torch.stack([prep_image(frame, args.img_size)[0] for frame in frame_list]).to(device)
vid_base_name = os.path.splitext(os.path.basename(args.vid_file))[0]
vid_feats = np.load(os.path.join(args.data_dir, args.corpus, 'bbox_feats/', vid_base_name + '.npy'))
vid_feats = torch.Tensor(vid_feats).to(device).unsqueeze(0)
with torch.no_grad():
    logits, seq_alphas = spatial_model(vid_feats)
    detections = yolo_model(frame_tensor, torch.cuda.is_available())
    pdb.set_trace()

seq_alphas = seq_alphas.squeeze(0).cpu().numpy()
grid_size = seq_alphas.shape[1]
n = args.img_size // grid_size
seq_alphas = seq_alphas.repeat(n, axis=1).repeat(n, axis=2)

vwriter = cv2.VideoWriter('tmp.mp4', \
    0x7634706d, \
    2, (args.img_size, args.img_size))

for i, frame in enumerate(frame_list):
    img_tensor, img = prep_image(frame, args.img_size)
    att = seq_alphas[i]
    att = (att - att.min()) / (att.max() - att.min())
    att = (att / 2.0) + 0.5
    img = np.multiply(img, np.expand_dims(att, axis=2))
    vwriter.write(np.uint8(img))

vwriter.release()

