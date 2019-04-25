import pdb
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
sys.path.append('..')
from utils import *


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * hidden_size
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.hidden_size)
        
        output = self.out(concat)
    
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(hidden_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, hidden_size)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
    
        self.size = 4096#hidden_size
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        #print ("x", x.shape)
        #print ("x.mean(dim=-1, keepdim=True)", x.mean(dim=-1, keepdim=True).shape)
        #print ("x.std(dim=-1, keepdim=True)",x.std(dim=-1, keepdim=True).shape)
        #print ("self.bias", self.bias.shape)
        #print ("(x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True)", ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True))).shape)
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(hidden_size)
        self.norm_2 = Norm(hidden_size)
        self.attn = MultiHeadAttention(heads, hidden_size)
        self.ff = FeedForward(hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        print ("x2 shape", x2.shape)
        x = x + self.dropout_1(self.attn(x2,x2,x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(hidden_size)
        self.norm_2 = Norm(hidden_size)
        self.norm_3 = Norm(hidden_size)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, hidden_size)
        self.attn_2 = MultiHeadAttention(heads, hidden_size)
        self.ff = FeedForward(hidden_size).cuda()
    def forward(self, x, e_outputs):#, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2))#, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs))#,src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, N, heads):
        super().__init__()
        self.N = N
        #self.embed = Embedder(vocab_size, hidden_size)
        #self.pe = PositionalEncoder(hidden_size)
        self.layers = get_clones(EncoderLayer(hidden_size, heads), N)
        self.norm = Norm(hidden_size)
    def forward(self, vid_features):
        #x = self.embed(src)
        #x = self.pe(x)
        #vid_feats = vid_feats.transpose(0, 1)
        # B x N x V
        for i in range(self.N):
            x = self.layers[i](vid_features)
        # returning encoder_outs
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, glove_loader, hidden_size, dropout_p,max_len, N, heads):
        super().__init__()
        self.N = N
        word_vectors = glove_loader.word_vectors
        word_vectors = np.vstack(word_vectors)
        self.vocab_size = word_vectors.shape[0]
        self.embed_size = word_vectors.shape[1]
        self.max_len = max_len

        self.sos_id = glove_loader.get_id('<sos>')

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})

        #self.embed = Embedder(vocab_size, hidden_size)
        #self.pe = PositionalEncoder(hidden_size)
        self.layers = get_clones(DecoderLayer(hidden_size + self.embed_size, heads), N)
        self.norm = Norm(hidden_size)
        self.pred_linear = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(hidden_size, self.vocab_size))

    def forward(self, e_outputs,s):
        if self.training:
            assert s is not None
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

        batch_size = self.e_outputs.shape[0]
        num_frames = self.e_outputs.shape[1]
        sos_tags = torch.ones((batch_size, 1), dtype=torch.long) * self.sos_id
        sos_tags = sos_tags.to(device)
        if self.training:
            s = torch.cat((sos_tags, s), dim=1)
            # right shifted sentence - (B x (L + 1))
        else:
            s = sos_tags
        curr_words = s[:,0]
        #x = self.embed(curr_words)
        #x = self.pe(x)
        logits = None
        for i in range(self.max_len):
            for j in range(self.N):
                x = self.layers[j](curr_words, e_outputs)#, src_mask, trg_mask)
            outs = self.norm(x)
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




class Transformer(nn.Module):
    def __init__(self, glove_loader, dropout_p, hidden_size,vid_feat_size, max_len, N, heads):
        super().__init__()
        self.encoder = Encoder(vid_feat_size, hidden_size, N, heads)
        self.decoder = Decoder(glove_loader, hidden_size, dropout_p,max_len,N, heads)
        #self.out = nn.Linear(hidden_size, glove_loader)
    def forward(self, vid_features,s=None):# __trg__, __src_mask__, __trg_mask__):
        print ("vid_features shape:", vid_features.shape)
        e_outputs = self.encoder(vid_features, __src_mask__)
        logits = self.decoder(e_outputs,s)#__src_mask__, __trg_mask__)
        #logits = self.out(d_output)
        return logits
