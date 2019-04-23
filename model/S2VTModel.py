import pdb
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('..')
from utils import *

class S2VTModel(nn.Module):
    """Sequence to Sequence -- Video to Text model from
    https://arxiv.org/pdf/1505.00487.pdf"""
    def __init__(self, glove_loader, dropout_p, hidden_size, vid_feat_size, max_len):
        """
        Args:
            glove_loader: GLoVe embedding loader
            dropout_p: Dropout probability for intermediate dropout layers
            hidden_size: Size of the intermediate linear layers
            vid_feat_size: Size of the video features
            max_len: Max length to rollout
        """
        super(S2VTModel, self).__init__()

        word_vectors = glove_loader.word_vectors
        word_vectors = np.vstack(word_vectors)
        self.vocab_size = word_vectors.shape[0]
        self.embed_size = word_vectors.shape[1]
        self.vid_feat_size = vid_feat_size

        self.max_len = max_len
        self.sos_id = glove_loader.get_id('<sos>')

        self.teacher_force_prob = 1.0

        self.embedding = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_size),
            nn.Dropout(p=dropout_p))
        self.embedding[0].load_state_dict({'weight': torch.Tensor(word_vectors)})

        self.rnn1 = nn.GRU(input_size=vid_feat_size, hidden_size=hidden_size, \
            num_layers=1)
        self.rnn2 = nn.GRU(input_size=hidden_size + self.embed_size, hidden_size=hidden_size, \
            num_layers=1)

        self.linear = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, self.vocab_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize network weights using Xavier init (with bias 0.01)"""
        self.apply(ixvr)

    def encode_step(self, vid_feat, rnn_state=None):
        """
        Args:
            vid_feat: Video features (B x V)
            rnn_state: None if the beginning otherwise the hidden state
                of rnn1
        Output:
            output: Encoder output (1 x B x H)
            rnn_state: Next RNN state
        """
        if rnn_state is not None:
            output, rnn_state = self.rnn1(vid_feat.unsqueeze(0), rnn_state)
        else:
            output, rnn_state = self.rnn1(vid_feat.unsqueeze(0))

        return output, rnn_state

    def encode(self, vid_feats):
        """
        Args:
            vid_feats: Video features (B x N x V)
        Output:
            output: Encoder output (N x B x H)
            rnn_state: Next RNN state
        """
        vid_feats = torch.transpose(vid_feats, 0, 1)
        # N x B x V
        output, rnn_state = self.rnn1(vid_feats)

        return output, rnn_state

    def decode(self, output1, state1, s):
        """
        Args:
            output1: rnn1 output (N x B x H)
            state1: rnn1 state (1 x B x H)
            s: Tokenized sentence (B x L)
        Output:
            logits: Logits for next word prediction (B x L x vocab_size)
        """
        device = output1.device

        num_frames = output1.shape[0]
        batch_size = output1.shape[1]
        max_len = self.max_len

        word_padding = torch.zeros((num_frames, batch_size, self.embed_size)).to(device)
        # N x B x E
        output1 = torch.cat((output1, word_padding), dim=2)
        # N x B x (H + E)
        _, state2 = self.rnn2(output1)

        if self.training:
            # uses teacher forcing with scheduled sampling
            vid_feats = torch.zeros((1, batch_size, self.vid_feat_size)).to(device)
            # 1 x B x V
            sos_tags = torch.ones((batch_size, 1), dtype=torch.long) * self.sos_id
            sos_tags = sos_tags.to(device)
            s = torch.cat((sos_tags, s), dim=1)
            # right shifted sentence - (B x (L + 1))
            curr_words = s[:, 0].unsqueeze(0)
            # 1 x B
            logits = None

            for i in range(max_len):
                output1, state1 = self.rnn1(vid_feats, state1)
                # 1 x B x H
                curr_words = self.embedding(curr_words)
                # 1 x B x E
                output1 = torch.cat((output1, curr_words), dim=2)
                # 1 x B x (H + E)
                output2, state2 = self.rnn2(output1, state2)
                # 1 x B x H
                outs = self.linear(output2.squeeze(0))
                _, curr_words = torch.max(outs, dim=1)
                curr_words = curr_words.unsqueeze(0)
                # 1 x B
                if random.random() < self.teacher_force_prob:
                    # with teacher forcing prob keep the teacher word
                    curr_words = s[:, i + 1].unsqueeze(0)
                # 1 x B
                if logits is None:
                    logits = outs.unsqueeze(0)
                else:
                    logits = torch.cat((logits, outs.unsqueeze(0)), dim=0)

            logits = torch.transpose(logits, 0, 1).contiguous()
            # B x L x vocab_size

            return logits
        else:
            # need to do rollouts
            vid_frames = torch.zeros((1, batch_size, self.vid_feat_size)).to(device)
            # 1 x B x V
            curr_words = torch.ones((1, batch_size), dtype=torch.long) * self.sos_id
            curr_words = curr_words.to(device)
            # 1 x B
            logits = None

            for i in range(max_len):
                output1, state1 = self.rnn1(vid_frames, state1)
                # 1 x B x H
                curr_words = self.embedding(curr_words)
                # 1 x B x E
                output1 = torch.cat((output1, curr_words), dim=2)
                # 1 x B x (H + E)
                output2, state2 = self.rnn2(output1, state2)
                # 1 x B x H
                outs = self.linear(output2.squeeze(0))
                _, curr_words = torch.max(outs, dim=1)
                curr_words = curr_words.unsqueeze(0)
                # 1 x B
                if logits is None:
                    logits = outs.unsqueeze(0)
                else:
                    logits = torch.cat((logits, outs.unsqueeze(0)), dim=0)

            logits = torch.transpose(logits, 0, 1).contiguous()
            # B x L x vocab_size
            
            return logits

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
        """
        if self.training:
            assert s is not None

        output1, state1 = self.encode(vid_feats)
        # output1 - (N x B x H)
        logits = self.decode(output1, state1, s)

        return logits

