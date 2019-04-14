import os
import pdb
import time
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from nlgeval import NLGEval
from model.S2VTModel import S2VTModel
from logger import TensorboardXLogger
from dataset import MSVideoDescriptionDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
    mask = torch.FloatTensor(mask)

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
    loss = torch.mul(loss, mask).sum() / torch.sum(mask)

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
    correct = pred.eq(target)
    mask = calc_sentence_mask(batch_size, max_len, s_len)
    acc = torch.mul(correct, mask).sum() / torch.sum(mask)

    return acc

def calc_meteor_score(pred, refs, glove_loader, nlg_eval):
    """
    Args:
        pred: output predictions of the classifier (B x L)
        refs: list of lists containing the reference sentences
        glove_loader: GloveLoader class object
        nlg_eval: NLGEval class object
    Output:
        score: avg. METEOR score for the batch
    """
    hyps = glove_loader.get_sents_from_indexes(pred.data.cpu().numpy())
    score = 0.0
    for ref, p in zip(refs, hyps):
        score += nlg_eval.compute_individual_metrics(ref=ref, hyp=hyp)['METEOR']
    score /= pred.shape[0]

    return score

def run_iter(opts, data, model, criterion, return_pred=False):
    vid_feats, s, s_len = data['vid_feats'].to(device), data['sent'].to(device), data['sent_len'].to(device)
    logits = model(vid_feats, s)
    pred = torch.argmax(logits, dim=2)
    loss = calc_masked_loss(logits, s, s_len, criterion)
    acc = calc_masked_accuracy(logits, target, s_len)

    if not return_pred:
        return acc, loss
    return acc, loss, pred

def evaluate(opts, model, loader, criterion, meteor_eval_func):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_acc = 0.0
    val_meteor_score = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            acc, loss, pred = run_iter(opts, data, model, criterion, return_pred=True)
            meteor_score = meteor_eval_func(pred, data['refs'])
            val_loss += loss.data.cpu().item()
            val_acc += acc
            val_meteor_score += meteor_score
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_acc = val_acc / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_acc, val_meteor_score, time_taken


def train(opts):

    glove_loader = GloveLoader(os.path.join(opts.data_dir, 'glove/trunc', opts.glove_emb_file))
    train_loader = DataLoader(MSVideoDescriptionDataset(opts.data_dir, 'train', glove_loader, opts.num_frames, opts.max_len), \
        batch_size=opts.bsize, shuffle=True, num_workers=opts.nworkers)
    valid_loader = DataLoader(MSVideoDescriptionDataset(opts.data_dir, 'val', glove_loader, opts.num_frames, opts.max_len), \
        batch_size=opts.bsize, shuffle=False, num_workers=opts.nworkers)

    if opts.arch == 's2vt':
        model = S2VTModel(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len)
    else:
        raise NotImplementedError('Unknown model architecture')

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)
    else:
        raise NotImplementedError("Unknown optim type")

    criterion = nn.CrossEntropyLoss(reduction='none')
    metrics_to_omit = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', \
        'ROUGE_L', 'CIDEr', 'SkipThoughtCS', \
        'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', \
        'GreedyMatchingScore']
    nlg_eval = NLGEval(metrics_to_omit=metrics_to_omit)

    start_n_iter = 0
    # for choosing the best model
    best_val_meteor_score = 0.0

    model_path = os.path.join(opts.save_path, 'model_latest.net')
    if opts.resume and os.path.exists(model_path):
        # restoring training from save_state
        print ('====> Resuming training from previous checkpoint')
        save_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(save_state['state_dict'])
        start_n_iter = save_state['n_iter']
        best_val_meteor_score = save_state['best_val_meteor_score']
        opts = save_state['opts']
        opts.start_epoch = save_state['epoch'] + 1

    model = model.to(device)

    # for logging
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['acc', 'loss'])
    logger.n_iter = start_n_iter

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        logger.step()

        for batch_idx, data in enumerate(train_loader):
            acc, loss = run_iter(opts, data, model, criterion)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(acc, loss)

        meteor_eval_func = lambda pred, refs: calc_meteor_score(pred, refs, glove_loader, nlg_eval)
        val_loss, val_acc, val_meteor_score, time_taken = evaluate(opts, model, valid_loader, criterion, meteor_eval_func)
        # log the validation losses
        logger.log_valid(time_taken, val_acc, val_loss)
        logger.writer.add_scalar('val/METEOR', val_meteor_score, logger.n_iter)
        print ('')

        # Save the model to disk
        if val_meteor_score >= best_val_meteor_score:
            best_val_meteor_score = val_meteor_score
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
                'opts': opts,
                'val_meteor_score': val_meteor_score,
                'best_val_meteor_score': best_val_meteor_score
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_meteor_score': val_meteor_score,
            'best_val_meteor_score': best_val_meteor_score
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)

