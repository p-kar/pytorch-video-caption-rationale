import os
import csv
import pdb
import json
import numpy as np
import random
from nltk import word_tokenize

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def read_caption_file(fname):
    """
    Args:
        fname: file containing the video captions
    Output:
        samples: dict containing the captions and the metadata
    """
    with open(fname, 'r') as fp:
        samples = json.load(fp)
    for sample in samples:
        for i in range(len(sample['captions'])):
            sample['captions'][i]['desc'] = word_tokenize(sample['captions'][i]['desc'])
    return samples

def collate_fn(batch):
    """
    default collate function in PyTorch handles lists weirdly
    """
    if isinstance(batch[0], dict):
        ret_dict = {}
        for k in batch[0]:
            if k == 'refs':
                ret_dict[k] = [b[k] for b in batch]
            else:
                ret_dict[k] = default_collate([b[k] for b in batch])
        return ret_dict
    return default_collate(batch)

class MSVideoDescriptionDataset(Dataset):
    """Microsoft Video Description Corpus"""

    def __init__(self, root, corpus, split, glove_loader, num_frames, maxlen):
        assert(corpus in ['msvd', 'msvd_vgg'])
        self.word_to_index = glove_loader.word_to_index
        self.split = split
        self.glove_vec_size = glove_loader.embed_size
        self.corpus_dir = os.path.join(root, corpus)
        self.caption_file = os.path.join(self.corpus_dir, '{}_captions.json'.format(split))
        self.captions = read_caption_file(self.caption_file)
        self.maxlen = maxlen
        self.num_frames = num_frames
        self.vid_feat_dir = os.path.join(self.corpus_dir, 'feats/')

    def __len__(self):
        return len(self.captions)

    def _parse(self, sent):
        sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
        sent.append('<eos>')
        sent = sent[:self.maxlen]
        padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
        sent.extend(padding)
        return np.array([self.word_to_index[s] for s in sent])

    def __getitem__(self, idx):
        # Extract ImageNet features for video frames
        video_key = self.captions[idx]['video_key']
        vid_feats = np.load(os.path.join(self.vid_feat_dir, video_key + '.npy'))
        vid_feats_padding = np.zeros((max(0, self.num_frames - vid_feats.shape[0]), vid_feats.shape[1]))
        vid_feats = np.concatenate((vid_feats, vid_feats_padding), axis=0)[:self.num_frames, :]
        vid_feats = torch.FloatTensor(vid_feats)
        # Get a random caption for this video
        sent_toks = random.choice(self.captions[idx]['captions'])['desc']
        sent_raw = ' '.join(sent_toks).lower()
        sent = torch.LongTensor(self._parse(sent_toks))
        sent_len = min(self.maxlen, len(sent_toks) + 1)
        # Reference sentences for non-training splits
        # if self.split != 'train':
        refs = np.array([' '.join(cap['desc']).lower() for cap in self.captions[idx]['captions']])

        return {'sent': sent, 'sent_raw': sent_raw, 'sent_len': sent_len, 'vid_feats': vid_feats, 'refs': refs, 'vid_key': video_key}

class MSRVideoToTextDataset(Dataset):
    """Microsoft Research - Video To Text Dataset"""

    def __init__(self, root, split, glove_loader, num_frames, maxlen):
        assert(corpus == 'msrvtt')
        self.word_to_index = glove_loader.word_to_index
        self.split = split
        self.glove_vec_size = glove_loader.embed_size
        self.corpus_dir = os.path.join(root, corpus)
        self.caption_file = os.path.join(self.corpus_dir, '{}_captions.json'.format(split))
        self.captions = read_caption_file(self.caption_file)
        self.maxlen = maxlen
        self.num_frames = num_frames
        self.vid_feat_dir = os.path.join(self.corpus_dir, 'feats/')

    def __len__(self):
        return len(self.captions)

    def _parse(self, sent):
        sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
        sent.append('<eos>')
        sent = sent[:self.maxlen]
        padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
        sent.extend(padding)
        return np.array([self.word_to_index[s] for s in sent])

    def __getitem__(self, idx):
        # Extract ImageNet features for video frames
        video_key = self.captions[idx]['video_id']
        vid_feats = np.load(os.path.join(self.vid_feat_dir, video_key + '.npy'))
        vid_feats_padding = np.zeros((max(0, self.num_frames - vid_feats.shape[0]), vid_feats.shape[1]))
        vid_feats = np.concatenate((vid_feats, vid_feats_padding), axis=0)[:self.num_frames, :]
        vid_feats = torch.FloatTensor(vid_feats)
        # Get a random caption for this video
        sent_toks = random.choice(self.captions[idx]['captions'])['desc']
        sent_raw = ' '.join(sent_toks).lower()
        sent = torch.LongTensor(self._parse(sent_toks))
        sent_len = min(self.maxlen, len(sent_toks) + 1)
        # Reference sentences for non-training splits
        # if self.split != 'train':
        refs = np.array([' '.join(cap['desc']).lower() for cap in self.captions[idx]['captions']])

        return {'sent': sent, 'sent_raw': sent_raw, 'sent_len': sent_len, 'vid_feats': vid_feats, 'refs': refs, 'vid_key': video_key}

