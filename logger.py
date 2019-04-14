import os
import pdb
import sys
import time
import torch
import tensorboardX
from tensorboardX import SummaryWriter

class TensorboardXLogger:
    def __init__(self, start_epoch, log_iter, log_dir):
        self.log_iter = log_iter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.n_iter = 0
        self.epoch = start_epoch - 1
        self.time_start = time.time()
        self.num_batches = 0.0
        self.log_dict = {}
        self.log_keys = []

    def set(self, log_keys):
        self.log_keys = log_keys
        self.reset()

    def reset(self):
        self.num_batches = 0.0
        self.time_start = time.time()
        self.log_dict = {k: 0.0 for k in self.log_keys}

    def step(self):
        self.epoch += 1

    def update(self, *vals):
        vals = list(vals)
        vals = [v.data.cpu().item() if isinstance(v, torch.Tensor) else v for v in vals]
        assert len(vals) == len(self.log_keys)

        for k, v in zip(self.log_keys, vals):
            self.log_dict[k] += v
        self.n_iter += 1
        self.num_batches += 1

        if self.num_batches != 0 and self.n_iter % self.log_iter == 0:
            self.log_train()

    def log_train(self):
        assert self.num_batches != 0
        time_taken = time.time() - self.time_start

        for k in self.log_keys:
            self.log_dict[k] /= self.num_batches
            self.writer.add_scalar('train/' + k, self.log_dict[k], self.n_iter)

        values = [self.log_dict[k] for k in self.log_keys]
        self.reset()

        log_str = 'epoch: %d, updates: %d, time: %.2f, ' + ', '.join(['train_' + k + ': %.5f' for k in self.log_keys])
        print (log_str % (self.epoch, self.n_iter, time_taken, *values))

    def log_valid(self, time_taken, *vals):
        self.time_start += time_taken
        vals = list(vals)
        vals = [v.data.cpu().item() if isinstance(v, torch.Tensor) else v for v in vals]
        assert len(vals) == len(self.log_keys)

        for k, v in zip(self.log_keys, vals):
            self.writer.add_scalar('val/' + k, v, self.n_iter)

        log_str = 'epoch: %d, updates: %d, time: %.2f, ' + ', '.join(['val_' + k + ': %.5f' for k in self.log_keys])
        print (log_str % (self.epoch, self.n_iter, time_taken, *vals))
