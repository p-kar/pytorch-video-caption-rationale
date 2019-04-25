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

class Attention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    def __init__(self, hidden_size):
        """
        Args:
            hidden_size: size of the hidden layer
        """
        super(Attention, self).__init__()

        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, feats):
        """
        Args:
            query: hidden state of the encoder (B x H)
            key: keys (B x N x H)
            feats: spatial features (B x N x V)
        Output:
            context: attention combined spatial features (B x V)
        """
        batch_size, hidden_size = query.shape

        proj_key = self.key_layer(key.contiguous().view(-1, hidden_size)).view(batch_size, -1, hidden_size)
        # B x N x H
        query = self.query_layer(query)
        # B x H
        energy_input = torch.tanh(query.unsqueeze(1) + proj_key).view(-1, hidden_size)
        # (B * N) x H
        scores = self.energy_layer(energy_input).view(batch_size, -1)
        # B x N
        # we do not do any masking here because we assume all the feats are valid
        alphas = F.softmax(scores, dim=1)
        # B x N
        context = torch.bmm(alphas.unsqueeze(1), feats).squeeze(1)
        # B x V

        return context

class SpatialNet(nn.Module):
    """Spatial attention networks using YOLOv3 as backbone"""
    def __init__(self, glove_loader, dropout_p, hidden_size, vid_feat_size, max_len, arch):
        """
        Args:
            glove_loader: GLoVe embedding loader
            dropout_p: Dropout probability for intermediate dropout layers
            hidden_size: Size of the intermediate linear layers
            vid_feat_size: Size of the video features
            max_len: Max length to rollout
            arch: video captioning network ['s2vt' | 's2vt-att']
        """
        super(SpatialNet, self).__init__()

        if arch == 's2vt':
            self.caption_net = S2VTModel(glove_loader, dropout_p, hidden_size, vid_feat_size, max_len)
        else:
            raise NotImplementedError('unknown video captioning arch')

        self.conv = nn.Sequential(
            nn.Conv2d(vid_feat_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU())

        self.attention = Attention(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, vid_feats, s=None):
        """
        Args:
            vid_feats: Video features (B x N x F x K x K)
            s: Tokenized sentence (B x L)
                sentence doesn't start with <sos> but ends 
                with <eos> (with optional <pad>)
        Output:
            logits: Logits for next word prediction (B x L x vocab_size)
        """
        batch_size = vid_feats.shape[0]
        num_frames = vid_feats.shape[1]
        num_filters = vid_feats.shape[2]
        grid_size = vid_feats.shape[3]
        num_cells = grid_size * grid_size
        device = vid_feats.device

        conv_feats = self.conv(vid_feats.view(-1, num_filters, grid_size, grid_size))
        conv_feats = conv_feats.view(batch_size, num_frames, -1, num_cells)
        # B x N x F' x K^2
        conv_feats = torch.transpose(conv_feats, 2, 3)
        # B x N x K^2 x F'
        vid_feats = vid_feats.view(batch_size, num_frames, num_filters, -1)
        vid_feats = torch.transpose(vid_feats, 2, 3)
        # B x N x K^2 x F

        rnn_state = torch.zeros(1, batch_size, self.hidden_size).to(device)
        # 1 x B x H
        output1 = None

        for i in range(num_frames):
            frame_conv_feats = conv_feats[:,i,:,:]
            # B x K^2 x F'
            frame_vid_feats = vid_feats[:,i,:,:]
            # B x K^2 x F
            context = self.attention(rnn_state.squeeze(0), frame_conv_feats, frame_vid_feats)
            # B x F
            out, rnn_state = self.caption_net.encode_step(context, rnn_state)

            if output1 is None:
                output1 = out
            else:
                output1 = torch.cat((output1, out), dim=0)

        logits = self.caption_net.decode(output1, rnn_state, s)

        return logits

