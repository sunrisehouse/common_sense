# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from pdb import set_trace
from utils.tensor import convert_to_tensor

def make_dataloader(task, *args, **kwargs):
    if task == 'kor_common':
        return _make_dataloader_korkg(*args, **kwargs)

def _make_dataloader_korkg(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for i, example in enumerate(examples):
        features, la = example.fl(tokenizer, max_seq_length)

        one_hot = np.zeros(shape = (len(features), ), dtype=np.int8)
        one_hot = one_hot.tolist()
        one_hot[int(la)-1] = 1

        F.extend(features)
        L.extend(one_hot)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 