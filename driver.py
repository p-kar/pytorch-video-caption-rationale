import os
import pdb

from utils import *
from args import get_args
from train import train
from data.extract_glove import extract_glove
from data.extract_video_feats import extract_video_feats
from data.msvd.extract_captions import extract_captions as msvd_extract_captions
from data.msrvtt.extract_captions import extract_captions as msrvtt_extract_captions

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'extract_captions':
        if opts.corpus == 'msvd':
            msvd_extract_captions(opts)
        elif opts.corpus == 'msrvtt':
            msrvtt_extract_captions(opts)
        else:
            raise NotImplementedError('unknown corpus')
    elif opts.mode == 'extract_video_feats':
        extract_video_feats(opts)
    elif opts.mode == 'extract_glove':
        extract_glove(opts)
    elif opts.mode == 'train':
        train(opts)
    else:
        raise NotImplementedError('unrecognized mode')

