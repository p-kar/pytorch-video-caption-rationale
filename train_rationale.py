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
from dataset import *
from train_utils import *
from nlgeval import NLGEval
from model.RationaleNet import RationaleNet
from logger import TensorboardXLogger

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def run_iter(opts, data, model, criterion, return_pred=False):
    vid_feats, s, s_len = data['vid_feats'].to(device), data['sent'].to(device), data['sent_len'].to(device)
    logits, probs = model(vid_feats, s)
    pred = torch.argmax(logits, dim=2)
    loss_ce = calc_masked_loss(logits, s, s_len, criterion)
    loss_brev = calc_brevity_loss(probs) * opts.lambda_brev
    loss_cont = calc_cont_loss(probs) * opts.lambda_cont
    acc = calc_masked_accuracy(logits, s, s_len)

    loss = loss_ce + loss_brev + loss_cont

    if not return_pred:
        return acc, loss, loss_ce, loss_brev, loss_cont
    return acc, loss, loss_ce, loss_brev, loss_cont, pred

def evaluate(opts, model, loader, criterion, glove_loader, meteor_eval_func):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_loss_ce = 0.0
    val_loss_brev = 0.0
    val_loss_cont = 0.0
    val_acc = 0.0
    val_meteor_score = 0.0
    num_batches = 0.0
    sampler = StreamSampler(opts.n_sample_sent)

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            acc, loss, loss_ce, loss_brev, loss_cont, pred = run_iter(opts, data, model, criterion, return_pred=True)
            hyps = glove_loader.get_sents_from_indexes(pred.data.cpu().numpy())
            for hyp, ref, vk in zip(hyps, data['refs'], data['vid_key']):
                ref = random.choice(ref)
                sampler.add((hyp, ref, vk))
            meteor_score = meteor_eval_func(hyps, data['refs'])
            val_loss += loss.data.cpu().item()
            val_loss_ce += loss_ce.data.cpu().item()
            val_loss_brev += loss_brev.data.cpu().item()
            val_loss_cont += loss_cont.data.cpu().item()
            val_acc += acc
            val_meteor_score += meteor_score
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_loss_ce = val_loss_ce / num_batches
    avg_valid_loss_brev = val_loss_brev / num_batches
    avg_valid_loss_cont = val_loss_cont / num_batches
    avg_valid_acc = val_acc / num_batches
    avg_meteor_score = val_meteor_score / num_batches
    time_taken = time.time() - time_start
    sample_sent = sampler.get()

    return avg_valid_loss, avg_valid_loss_ce, avg_valid_loss_brev, \
        avg_valid_loss_cont, avg_valid_acc, avg_meteor_score, \
        sample_sent, time_taken


def train_rationale(opts):

    glove_loader = GloveLoader(os.path.join(opts.data_dir, opts.corpus, 'glove/', opts.glove_emb_file))
    if opts.corpus in ['msvd', 'msvd_vgg']:
        VDDataset = MSVideoDescriptionDataset
    elif opts.corpus == 'msrvtt':
        VDDataset = MSRVideoToTextDataset
    else:
        raise NotImplementedError('Unknown dataset')

    train_loader = DataLoader(VDDataset(opts.data_dir, opts.corpus, 'train', glove_loader, opts.num_frames, opts.max_len), \
        batch_size=opts.bsize, shuffle=True, num_workers=opts.nworkers, collate_fn=collate_fn)
    valid_loader = DataLoader(VDDataset(opts.data_dir, opts.corpus, 'val', glove_loader, opts.num_frames, opts.max_len), \
        batch_size=opts.bsize, shuffle=False, num_workers=opts.nworkers, collate_fn=collate_fn)

    model = RationaleNet(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len, opts.tau, opts.arch)

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
    logger.set(['acc', 'loss', 'loss_ce', 'loss_brev', 'loss_cont'])
    logger.n_iter = start_n_iter

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        logger.step()

        sampler = StreamSampler(opts.n_sample_sent)
        for batch_idx, data in enumerate(train_loader):
            acc, loss, loss_ce, loss_brev, loss_cont, pred = run_iter(opts, data, model, criterion, return_pred=True)
            hyps = glove_loader.get_sents_from_indexes(pred.data.cpu().numpy())
            for hyp, ref, vk in zip(hyps, data['refs'], data['vid_key']):
                ref = random.choice(ref)
                sampler.add((hyp, ref, vk))

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(acc, loss, loss_ce, loss_brev, loss_cont)

        meteor_eval_func = lambda pred, refs: calc_meteor_score(pred, refs, nlg_eval)
        val_loss, val_loss_ce, val_loss_brev, val_loss_cont, val_acc, val_meteor_score, sample_sent, time_taken = evaluate(opts, model, valid_loader, criterion, glove_loader, meteor_eval_func)
        print('')
        print('********************************** TRAIN **********************************')
        train_sample_sent = sampler.get()
        print_sample_sents(train_sample_sent)
        print('***************************************************************************')
        print('')
        print('*********************************** VAL ***********************************')
        # log the validation losses
        logger.log_valid(time_taken, val_acc, val_loss, val_loss_ce, val_loss_brev, val_loss_cont)
        logger.writer.add_scalar('val/METEOR', val_meteor_score, logger.n_iter)
        print('Validation METEOR score: {:.5f}'.format(val_meteor_score))
        print_sample_sents(sample_sent)
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

