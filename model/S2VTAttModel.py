import pdb
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('..')
from utils import *

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

    def forward(self, query, proj_key, key):
        """
        Args:
            query: hidden state of the decoder (B x H)
            proj_key: projected keys (of the encoder states) precomputed
                for efficiency (B x N x H)
            key: keys (encoder states) (B x N x H)
        Output:
            context: attention combined encoder state (B x H)
        """
        batch_size, hidden_size = query.shape
        query = self.query_layer(query)
        # B x H
        energy_input = torch.tanh(query.unsqueeze(1) + proj_key).view(-1, hidden_size)
        # (B * N) x H
        scores = self.energy_layer(energy_input).view(batch_size, -1)
        # B x N
        # we do not do any masking here because we assume all the frames are valid
        alphas = F.softmax(scores, dim=1)
        # B x N
        context = torch.bmm(alphas.unsqueeze(1), key).squeeze(1)
        # B x H

        return context

class Encoder(nn.Module):
    """Encodes a sequence of video frames"""
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: input size of the video features
            hidden_size: output size of the RNN
        """
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, \
            num_layers=1)

    def forward(self, vid_feats):
        """
        Args:
            vid_feats: Video features (B x N x V)
        Output:
            encoder_outs: Encoder outputs (B x N x H)
            encoder_final: Encoder state final (1 x B x H)
        """
        vid_feats = vid_feats.transpose(0, 1)
        # N x B x V
        encoder_outs, encoder_final = self.rnn(vid_feats)
        # encoder_outs - N x B x H
        # encoder_final - 1 x B x H
        encoder_outs = encoder_outs.transpose(0, 1)
        # B x N x H

        return encoder_outs, encoder_final

class Decoder(nn.Module):
    """Decodes the video caption"""
    def __init__(self, glove_loader, hidden_size, dropout_p, max_len):
        """
        """
        super(Decoder, self).__init__()

        word_vectors = glove_loader.word_vectors
        word_vectors = np.vstack(word_vectors)
        self.vocab_size = word_vectors.shape[0]
        self.embed_size = word_vectors.shape[1]
        self.max_len = max_len

        self.sos_id = glove_loader.get_id('<sos>')

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embed_size.load_state_dict({'weight': torch.Tensor(word_vectors)})

        self.rnn = nn.LSTM(hidden_size + self.embed_size, hidden_size, \
            num_layers=1)

        self.attention = Attention(hidden_size=hidden_size)

        self.pred_linear = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, self.vocab_size))

    def forward_step(self, encoder_outs, proj_key, rnn_state, words):
        """
        Args:
            encoder_outs: Encoder outputs (B x N x H)
            proj_key: Projected encoder outputs (B x N x H)
            rnn_state: Decoder hidden state (1 x B x H)
            words: previous word for next decoder output (B)
        Output:
            logits: logits for next word prediction (B x vocab_size)
            rnn_state: Decoder next hidden state (1 x B x H)
        """
        context = self.attention(rnn_state.squeeze(0), proj_key, encoder_outs)
        # B x H
        words = self.embedding(words)
        # B x E
        decoder_inp = torch.cat((context, words), dim=1).unsqueeze(0)
        # 1 x B x (H + E)
        out, rnn_state = self.rnn(decoder_inp, rnn_state)
        # out - 1 x B x H
        # rnn_state - 1 x B x H
        out = self.pred_linear(out.squeeze(0))
        # B x vocab_size

        return out, rnn_state

    def forward(self, encoder_outs, encoder_final, s):
        """
        Args:
            encoder_outs: Encoder outputs (B x N x H)
            encoder_final: Encoder state final (1 x B x H)
            s: sentence tokens (B x L) or None if inferencing
        Output:
            logits: logits for the full sentence (B x L x vocab_size)
        """
        if self.training:
            assert s is not None
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

        batch_size = self.encoder_outs.shape[0]
        num_frames = self.encoder_outs.shape[1]

        sos_tags = torch.ones((batch_size, 1), dtype=torch.long) * self.sos_id
        sos_tags = sos_tags.to(device)
        if self.training:
            s = torch.cat((sos_tags, s), dim=1)
            # right shifted sentence - (B x (L + 1))
        else:
            s = sos_tags
        curr_words = s[:, 0]
        # B
        rnn_state = encoder_final
        # 1 x B x H
        proj_key = self.attention.key_layer(encoder_outs.view(batch_size * num_frames, -1)).view(batch_size, num_frames, -1)
        # B x N x H
        logits = None

        for i in range(self.max_len):
            outs, rnn_state = self.forward_step(encoder_outs, proj_key, rnn_state, curr_words)
            if logits is None:
                logits = outs.unsqueeze(0)
            else:
                logits = torch.cat((logits, outs.unsqueeze(0)), dim=0)
            if self.training:
                curr_words = s[:, i + 1]
            else:
                curr_words = torch.argmax(outs, dim=1)

        logits = torch.transpose(logits, 0, 1).contiguous()
        # B x L x vocab_size

        return logits


class S2VTAttModel(nn.Module):
    """S2VT model with Bahdanau attention"""
    def __init__(self, glove_loader, dropout_p, hidden_size, vid_feat_size, max_len):
        """
        Args:
            glove_loader: GLoVe embedding loader
            dropout_p: Dropout probability for intermediate dropout layers
            hidden_size: Size of the intermediate linear layers
            vid_feat_size: Size of the video features
            max_len: Max length to rollout
        """
        super(S2VTAttModel, self).__init__()

        self.encoder = Encoder(vid_feat_size, hidden_size)
        self.decoder = Decoder(glove_loader, hidden_size, dropout_p, max_len)

    def reset_parameter(self):
        """Initialize network weights using Xavier init (with bias 0.01)"""
        self.apply(ixvr)

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
        encoder_outs, encoder_final = self.encoder(vid_feats)
        logits = self.decoder(encoder_outs, encoder_final, s)

        return logits


