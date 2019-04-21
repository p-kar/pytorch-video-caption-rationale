import pdb
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('..')
from utils import *
from .S2VTModel import S2VTModel
from .S2VTAttModel import S2VTAttModel

class Generator(nn.Module):
    
    def __init__(self, dropout_p, hidden_size, vid_feat_size, tau):
        """
        Args:
            dropout_p: Dropout probability for intermediate dropout layers
            hidden_size: Size of the intermediate linear layers
            vid_feat_size: Size of the video features
            tau: non-negative scalar temperature
        """
        super(Generator, self).__init__()

        self.rnn = nn.LSTM(input_size=vid_feat_size, hidden_size=hidden_size, \
            bidirectional=True, num_layers=1)
        self.drop = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(hidden_size * 2, 2)
        self.tau = tau

    def forward(self, vid_feats):
        """
        Args:
            vid_feats: Video features (B x N x V)
        Output:
            sel_vid_feats: Selected video features (B x N x V)
            probs: Selection probability of the video frames (B x N x 2)
        """
        batch_size = vid_feats.shape[0]
        num_frames = vid_feats.shape[1]

        vid_feats = torch.transpose(vid_feats, 0, 1)
        out, _ = self.rnn(vid_feats)
        # N x B x (2 * H)
        out = self.drop(out).transpose(0, 1)
        # B x N x (2 * H)
        logits = self.linear(out.view(batch_size * num_frames, -1))
        # (B * N) x 2
        hard = False if self.training else True
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=hard).view(batch_size, num_frames, -1)
        # B x N x 2
        sel_vid_feats = vid_feats * probs[:, :, 1].unsqueeze(-1)
        # B x N x V
        return sel_vid_feats, probs


class RationaleNet(nn.Module):

    def __init__(self, glove_loader, dropout_p, hidden_size, vid_feat_size, max_len, tau, arch):
        """
        Args:
            glove_loader: GLoVe embedding loader
            dropout_p: Dropout probability for intermediate dropout layers
            hidden_size: Size of the intermediate linear layers
            vid_feat_size: Size of the video features
            max_len: Max length to rollout
            tau: non-negative scalar temperature
            arch: video captioning network ['s2vt' | 's2vt-att']
        """
        super(RationaleNet, self).__init__()

        if arch == 's2vt':
            self.caption_net = S2VTModel(glove_loader, dropout_p, hidden_size, vid_feat_size, max_len)
        elif arch == 's2vt-att':
            self.caption_net = S2VTAttModel(glove_loader, dropout_p, hidden_size, vid_feat_size, max_len)
        else:
            raise NotImplementedError('unknown video captioning arch')

        self.gen = Generator(dropout_p, hidden_size, vid_feat_size, tau)

    def forward(self, vid_feats, s=None):
        """
        B - batch size
        H - hidden size
        N - number of frames
        V - video feature size
        L - length of sentence
        E - word embed size
        Args:
            vid_feats: Video features (B x N x V)
            s: Tokenized sentence (B x L)
                sentence doesn't start with <sos> but ends 
                with <eos> (with optional <pad>)
        Output:
            logits: Logits for next word prediction (B x L x vocab_size)
            probs: Selection probability of the video frames (B x N x 2)
        """
        sel_vid_feats, probs = self.gen(vid_feats)
        logits = self.caption_net(vid_feats, s)

        return logits, probs

