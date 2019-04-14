import pdb
import sys
import torch
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
        vocab_size = word_vectors.shape[0]
        embed_size = word_vectors.shape[1]

        self.max_len = max_len
        self.pad_id = glove_loader.get_id('<pad>')
        self.sos_id = glove_loader.get_id('<sos>')
        self.vid_pad = torch.Parameter(torch.randn(vid_feat_size))

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})

        self.rnn1 = nn.LSTM(input_size=vid_feat_size, hidden_size=hidden_size, \
            num_layers=1, dropout_p=dropout_p)
        self.rnn2 = nn.LSTM(input_size=hidden_size + embed_size, hidden_size=hidden_size, \
            num_layers=1, dropout_p=dropout_p)

        self.linear = nn.Linear(hidden_size, vocab_size)

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

        batch_size = vid_feats.shape[0]
        num_frames = vid_feats.shape[1]
        max_len = self.max_len

        vid_feats = torch.transpose(vid_feats, 0, 1)
        # N x B x V
        output1, state1 = self.rnn1(vid_feats)
        # output1 - (N x B x H)
        word_padding = torch.ones((num_frames, batch_size), dtype=torch.long) * self.pad_id
        # N x B
        word_padding = self.embedding(word_padding)
        # N x B x E
        output1 = torch.cat((output1, word_padding), dim=2)
        # N x B x (H + E)
        _, state2 = self.rnn2(output1)

        if self.training:
            # uses teacher forcing
            vid_feats = self.vid_pad.expand(max_len, batch_size, -1)
            # L x B x V
            output1, _ = self.rnn1(vid_feats, state1)
            # L x B x H
            sos_tags = torch.ones((batch_size, 1), dtype=torch.long) * self.sos_id
            s = torch.cat((sos_tags, s), dim=1)[:,:-1]
            # right shifted sentence - (B x L)
            s = self.embedding(s).transpose(0, 1)
            # L x B x E
            output1 = torch.cat((output1, s), dim=2)
            # L x B x (H + E)
            output2, _ = self.rnn2(output1, state2)
            # L x B x H
            output2 = torch.transpose(output2, 0, 1)
            # B x L x H
            logits = self.linear(output2.view(batch_size * max_len, -1))
            logits = logits.view(batch_size, max_len, -1)

            return logits
        else:
            # need to do rollouts
            vid_frames = self.vid_pad.expand(1, batch_size, -1)
            # 1 x B x V
            curr_words = torch.ones((1, batch_size), dtype=torch.long) * self.sos_id
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
                outs = self.linear(output2.squeeze())
                _, curr_words = torch.max(outs, dim=1)
                curr_words = curr_words.unsqueeze(0)
                # 1 x B
                if logits is None:
                    logits = outs.unsqueeze(0)
                else:
                    logits = torch.cat((logits, outs.unsqueeze(0)), dim=0)

            logits = torch.transpose(logits, 0, 1)
            # B x L x vocab_size
            
            return logits

