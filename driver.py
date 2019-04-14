import os
import pdb
from utils import *
from args import get_args
from data.extract_glove import extract_glove
from data.extract_captions import extract_captions
from data.extract_video_feats import extract_video_feats

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'extract_captions':
        extract_captions(opts)
    elif opts.mode == 'extract_video_feats':
        extract_video_feats(opts)
    elif opts.mode == 'extract_glove':
        extract_glove(opts)
    else:
        raise NotImplementedError('unrecognized mode')

