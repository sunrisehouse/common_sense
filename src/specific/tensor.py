# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.tensor import convert_to_tensor


def make_dataloader(task, *args, **kwargs):
    if task == 'kor_common':
        return _make_dataloader_korkg(*args, **kwargs)

def _make_dataloader_korkg(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 