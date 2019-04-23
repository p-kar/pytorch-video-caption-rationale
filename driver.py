import os
import pdb

from utils import *
from args import get_args
from train import train
from train_rationale import train_rationale
from train_spatial import train_spatial
from data.extract_glove import extract_glove
from data.extract_video_feats import extract_video_feats
from data.msvd_vgg.extract_video_feats import extract_video_feats as msvd_vgg_extract_video_feats
from data.msvd.extract_captions import extract_captions as msvd_extract_captions
from data.msrvtt.extract_captions import extract_captions as msrvtt_extract_captions
from data.msvd_vgg.extract_captions import extract_captions as msvd_vgg_extract_captions
from data.extract_bbox_feats import extract_bbox_feats

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'extract_captions':
        if opts.corpus == 'msvd':
            msvd_extract_captions(opts)
        elif opts.corpus == 'msrvtt':
            msrvtt_extract_captions(opts)
        elif opts.corpus == 'msvd_vgg':
            msvd_vgg_extract_captions(opts)
        else:
            raise NotImplementedError('unknown corpus')
    elif opts.mode == 'extract_video_feats':
        if opts.corpus == 'msvd_vgg':
            msvd_vgg_extract_video_feats(opts)
        else:
            extract_video_feats(opts)
    elif opts.mode == 'extract_bbox_feats':
        extract_bbox_feats(opts)
    elif opts.mode == 'extract_glove':
        extract_glove(opts)
    elif opts.mode == 'train':
        train(opts)
    elif opts.mode == 'train_rationale':
        train_rationale(opts)
    elif opts.mode == 'train_spatial':
        train_spatial(opts)
    else:
        raise NotImplementedError('unrecognized mode')

