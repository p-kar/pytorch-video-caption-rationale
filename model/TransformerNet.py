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
import math
from torch.autograd import Variable

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, flag, dropout = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads
        if flag == 'e':
           inp_size1 = inp_size2 = inp_size3 = 4096
          
        elif flag == 'd':
           inp_size1 = inp_size2 = inp_size3 = 300
        else:
           inp_size1 = 300
           inp_size2 = 4096
           inp_size3 = 4096
        self.q_linear = nn.Linear(inp_size1, hidden_size)
        self.v_linear = nn.Linear(inp_size2, hidden_size)
        self.k_linear = nn.Linear(inp_size3, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, inp_size1)
    
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
    def __init__(self, hidden_size,flag, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        if flag == 'e':
           size = 4096
        else:
           size = 300
        self.linear_1 = nn.Linear(size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, size)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, hidden_size, flag, eps = 1e-6):
        super().__init__()
        if flag == 'e':
            self.size = 4096#hidden_size
        elif flag == 'd':
            self.size = 300
        else:
            raise NotImplementedError()
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len = 80):
        super().__init__()
        self.hidden_size = hidden_size
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i in range(0, hidden_size, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/hidden_size)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/hidden_size)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.hidden_size)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, flag, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(hidden_size,flag)
        self.norm_2 = Norm(hidden_size,flag)
        self.attn = MultiHeadAttention(heads, hidden_size, flag)
        self.ff = FeedForward(hidden_size,flag)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, heads,flag, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(hidden_size,flag)
        self.norm_2 = Norm(hidden_size,flag)
        self.norm_3 = Norm(hidden_size,flag)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, hidden_size,flag)
        self.attn_2 = MultiHeadAttention(heads, hidden_size,'ed')
        self.ff = FeedForward(hidden_size,flag).cuda()
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, N, heads,flag):
        super().__init__()
        self.N = N
        #self.embed = Embedder(vocab_size, hidden_size)
        self.pe = PositionalEncoder(hidden_size)
        self.layers = get_clones(EncoderLayer(hidden_size, heads,flag), N)
        self.norm = Norm(hidden_size,flag)
    def forward(self, vid_features, mask):
        #x = self.embed(src)
        #
        #vid_feats = vid_feats.transpose(0, 1)
        # B x N x V
        x = vid_features
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        # returning encoder_outs
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, glove_loader, hidden_size, dropout_p,max_len, N, heads,flag):
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
        self.pe = PositionalEncoder(hidden_size)
        self.layers = get_clones(DecoderLayer(hidden_size, heads,flag), N)
        self.norm = Norm(hidden_size,flag)
        self.norm_out = Norm(hidden_size,flag)
        self.pred_linear = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(300, self.vocab_size))

    def forward(self, e_outputs, s, src_mask, trg_mask):
        if self.training:
            assert s is not None
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

        batch_size = e_outputs.shape[0]
        num_frames = e_outputs.shape[1]
        sos_tags = torch.ones((batch_size, 1), dtype=torch.long) * self.sos_id
        sos_tags = sos_tags.to(device)
        if not self.training:
           s = sos_tags
        logits = None

        if self.training:
            x = self.embedding(s)
            x = self.pe(x)
            for j in range(self.N):
                x = self.layers[j](x, e_outputs, src_mask, trg_mask)
            x = self.norm_out(x)
            logits = self.pred_linear(x)
        # B x L x vocab_size
        else:
            curr_words = s[:, 0]
            outputs = torch.zeros((batch_size, self.max_len)).long().to(device)
            logits = None
            for i in range(1, self.max_len+1):
                  outputs[:,i - 1] = curr_words
                  x = self.embedding(outputs[:,:i])
                  x = self.pe(x)
                  trg_mask = np.triu(np.ones((1, i, i)),k=1).astype('uint8')
                  trg_mask = (torch.from_numpy(trg_mask) == 0).to(device)
                  for j in range(self.N):
                      x = self.layers[j](x, e_outputs, src_mask, trg_mask)
                  x = self.norm_out(x)
                  outs = self.pred_linear(x[:,i - 1,:])
                  if logits is None:
                    logits = outs.unsqueeze(1)
                  else:
                    logits = torch.cat((logits, outs.unsqueeze(1)), dim=1)
                  curr_words = torch.argmax(outs, dim=1).long()

        return logits

def create_masks_inp(vid_features):
        input_msk = torch.ones(vid_features.shape[:2]).unsqueeze(1).to(vid_features.device)
        return input_msk

def create_masks_trg(self, s, s_len):
    if self.training:
        assert s is not None
        device = s.device
        trg_mask = torch.arange(s.shape[1]).expand_as(s).to(device)
        trg_mask = trg_mask < s_len.unsqueeze(1)
        trg_mask = trg_mask.to(device).unsqueeze(1)
        size = s.shape[1]

        nopeak_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
        nopeak_mask = (torch.from_numpy(nopeak_mask) == 0).to(device)
        trg_mask = trg_mask & nopeak_mask
    else:
        return None

    return trg_mask 

class Transformer(nn.Module):
    def __init__(self, glove_loader, dropout_p, hidden_size,vid_feat_size, max_len, N, heads):
        super().__init__()
        self.encoder = Encoder(vid_feat_size, hidden_size, N, heads,flag = 'e')
        self.decoder = Decoder(glove_loader, hidden_size, dropout_p, max_len, N, heads,flag ='d')

    def forward(self, vid_features, s=None, s_len=None):
        src_mask = create_masks_inp(vid_features)
        e_outputs = self.encoder(vid_features,src_mask)
        trg_mask = create_masks_trg(self, s, s_len)
        logits = self.decoder(e_outputs,s,src_mask,trg_mask)
        return logits
