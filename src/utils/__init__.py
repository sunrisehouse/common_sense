# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os
import random
import numpy as np

def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_device(gpu_ids):
    if gpu_ids:
        device_name = 'cuda:' + str(gpu_ids[0])
        n_gpu = torch.cuda.device_count()
        print('device is cuda, # cuda is: %d' % n_gpu)
    else:
        device_name = 'cpu'
        print('device is cpu')
    # print('!!!!!device_name=', device_name)
    device = torch.device(device_name)
    return device

class AvgVar:
    """
    accumlate average variable
    """
    def __init__(self):
        self.var = 0
        self.steps = 0

    def inc(self, v, step=1):
        self.var += v
        self.steps += step

    def avg(self):
        return self.var / self.steps if self.steps else 0


class Vn:
    """
    accumlate average variable list
    """
    def __init__(self, n):
        self.n = n
        self.vs = [AvgVar() for i in range(n)]

    def __getitem__(self, key):
        return self.vs[key]

    def init(self):
        self.vs = [AvgVar() for i in range(self.n)]

    def inc(self, vs):
        for v, _v in zip(self.vs, vs):
            v.inc(_v)

    def avg(self):
        return [v.avg() for v in self.vs]

    def list(self):
        return [v.var for v in self.vs]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def clip_batch(batch):
    """
    clip batch based on max length
    """
    # print("batch size is {}".format(len(batch[0])))
    idx, input_ids, attention_mask, token_type_ids, labels = batch
    # [batch_size, 2, L]
    batch_size = input_ids.size(0)
    while True:
        end_flag = False
        for i in range(batch_size):
            if input_ids[i, 0, -1] != 0:
                end_flag = True
            if input_ids[i, 1, -1] != 0:
                end_flag = True 
        
        if end_flag:
            break
        else:
            input_ids = input_ids[:, :, :-1]
    
    max_seq_length = input_ids.size(2)
    attention_mask = attention_mask[:, :, :max_seq_length]
    token_type_ids = token_type_ids[:, :, :max_seq_length]
    
    return idx, input_ids, attention_mask, token_type_ids, labels
