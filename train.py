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
from model.S2VTModel import S2VTModel
from model.S2VTAttModel import S2VTAttModel
from model.TransformerNet import Transformer 
from logger import TensorboardXLogger

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def run_iter(opts, data, model, criterion, return_pred=False):
    vid_feats, s, s_len = data['vid_feats'].to(device), data['sent'].to(device), data['sent_len'].to(device)
    if opts.arch == 'transformer':
       logits = model(vid_feats, s, s_len)
    else:
       logits = model(vid_feats,s)
    pred = torch.argmax(logits, dim=2)
    loss = calc_masked_loss(logits, s, s_len, criterion)
    acc = calc_masked_accuracy(logits, s, s_len)

    if not return_pred:
        return acc, loss
    return acc, loss, pred

def evaluate(opts, model, loader, criterion, glove_loader, meteor_eval_func):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_acc = 0.0
    val_meteor_score = 0.0
    num_batches = 0.0
    sampler = StreamSampler(opts.n_sample_sent)

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            acc, loss, pred = run_iter(opts, data, model, criterion, return_pred=True)
            hyps = glove_loader.get_sents_from_indexes(pred.data.cpu().numpy())
            for hyp, ref, vk in zip(hyps, data['refs'], data['vid_key']):
                ref = random.choice(ref)
                sampler.add((hyp, ref, vk))
            meteor_score = meteor_eval_func(hyps, data['refs'])
            val_loss += loss.data.cpu().item()
            val_acc += acc
            val_meteor_score += meteor_score
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_acc = val_acc / num_batches
    avg_meteor_score = val_meteor_score / num_batches
    time_taken = time.time() - time_start
    sample_sent = sampler.get()

    return avg_valid_loss, avg_valid_acc, avg_meteor_score, sample_sent, time_taken


def train(opts):

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

    if opts.arch == 's2vt':
        model = S2VTModel(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len)
    elif opts.arch == 's2vt-att':
        model = S2VTAttModel(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len)
    elif opts.arch == 'transformer':
        #inputs are number of layers and number of heads (last two)
        model = Transformer(glove_loader, opts.dropout_p, opts.hidden_size, opts.vid_feat_size, opts.max_len, 6, 8) 
    else:
        raise NotImplementedError('Unknown model architecture')

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)
    else:
        raise NotImplementedError("Unknown optim type")

    if opts.schedule_sample:
        sample_probs = inverse_sigmoid(opts.epochs)
    else:
        sample_probs = np.ones(opts.epochs)

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
        model.teacher_force_prob = sample_probs[epoch]
        logger.step()

        sampler = StreamSampler(opts.n_sample_sent)
        for batch_idx, data in enumerate(train_loader):
            acc, loss, pred = run_iter(opts, data, model, criterion, return_pred=True)
            hyps = glove_loader.get_sents_from_indexes(pred.data.cpu().numpy())
            for hyp, ref, vk in zip(hyps, data['refs'], data['vid_key']):
                ref = random.choice(ref)
                sampler.add((hyp, ref, vk))

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(acc, loss)

        meteor_eval_func = lambda pred, refs: calc_meteor_score(pred, refs, nlg_eval)
        val_loss, val_acc, val_meteor_score, sample_sent, time_taken = evaluate(opts, model, valid_loader, criterion, glove_loader, meteor_eval_func)
        print('')
        print('********************************** TRAIN **********************************')
        train_sample_sent = sampler.get()
        print_sample_sents(train_sample_sent)
        print('***************************************************************************')
        print('')
        print('*********************************** VAL ***********************************')
        # log the validation losses
        logger.log_valid(time_taken, val_acc, val_loss)
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

