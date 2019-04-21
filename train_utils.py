import pdb
import time
import torch
import random
import numpy as np
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def inverse_sigmoid(l, k = 140.0):
    """
    Args:
        l: size of the inverse sigmoid array desired
    Output:
        np.array of size l with the inverse sigmoid probs
        eg. for l = 1000 starts at 0.9929078 and ends at
        0.1002841
    """
    return k / (k + np.exp(np.arange(l) / k))

def calc_sentence_mask(batch_size, max_len, s_len):
    """
    Args:
        batch_size: batch size of the sentences
        max_len: maximum length of each sentence in a batch
        s_len: length of each sentence (B)
    Output:
        mask: 0/1 mask for the sentences based on the length (B x L)
    """
    mask = torch.arange(0, max_len).expand(batch_size, -1).to(device)
    mask = mask < s_len.unsqueeze(-1)
    mask = mask.float()

    return mask

def calc_masked_loss(logits, target, s_len, criterion):
    """
    Args:
        logits: output of the classifier (B x L x vocab_size)
        target: ground truth word labels (B x L)
        s_len: length of each sentence (B)
        criterion: cross entropy loss criterion (reduction = 'none')
    Output:
        loss: masked cross entropy loss
    """
    batch_size, max_len, _ = logits.shape
    loss = criterion(logits.view(batch_size * max_len, -1), target.view(-1))
    loss = loss.view(batch_size, max_len)
    mask = calc_sentence_mask(batch_size, max_len, s_len)
    loss = (torch.mul(loss, mask).sum(dim=1) / mask.sum(dim=1)).mean()
    # loss = torch.mul(loss, mask).sum() / torch.sum(mask)

    return loss

def calc_masked_accuracy(logits, target, s_len):
    """
    Args:
        logits: output of the classifier (B x L x vocab_size)
        target: ground truth word labels (B x L)
        s_len: length of each sentence (B)
    Output:
        acc: masked model accuracy
    """
    batch_size, max_len, _ = logits.shape
    pred = torch.argmax(logits, dim=2)
    correct = pred.eq(target).float()
    mask = calc_sentence_mask(batch_size, max_len, s_len)
    acc = torch.mul(correct, mask).sum() / torch.sum(mask)

    return acc

def calc_cont_loss(probs):
    """
    Args:
        probs: Selection probability for video frames (B x N x 2)
    Output:
        loss: Continuity loss |z_t - z_{t - 1}|
    """
    probs = probs[:, :, 1]
    # B x N
    loss = torch.mean(torch.abs(probs[:,1:] - probs[:,:-1]))
    return loss

def calc_brevity_loss(probs):
    """
    Args:
        probs: Selection probability for video frames (B x N x 2)
    Output:
        loss: Brevity loss (|z|)
    """
    probs = probs[:, :, 1]
    # B x N
    loss = torch.mean(torch.sum(probs, dim=1))
    return loss

def calc_meteor_score(hyps, refs, nlg_eval):
    """
    Args:
        hyps: list containing output prediction sentences of the classifier
        refs: list of lists containing the reference sentences
        nlg_eval: NLGEval class object
    Output:
        score: avg. METEOR score for the batch
    """
    score = 0.0
    for ref, hyp in zip(refs, hyps):
        score += nlg_eval.compute_individual_metrics(ref=ref, hyp=hyp)['METEOR']
    score /= len(hyps)

    return score

def print_sample_sents(tups):
    """
    Args:
        tuples containing
            hyps: output prediction sentences of the classifier
            refs: reference sentences
            vid_keys: video keys corresponding to the sentences
    """
    print('********************************* Samples *********************************')
    for hyp, ref, vk in tups:
        print('Video ID   : {}'.format(vk))
        print('Hypothesis : {}'.format(hyp))
        print('Reference  : {}'.format(ref))
        print('')
    print('***************************************************************************')

