# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertPreTrainedModel
from transformers import AlbertModel

import math
import logging

class AttentionMerge(nn.Module):
    """
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.
        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)

        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context

logger = logging.getLogger(__name__)

class Model(AlbertPreTrainedModel):
    """
    AlBert-AttentionMerge-Classifier

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """
    def __init__(self, config, no_att_merge=False, N_choices = 5):
        super(Model, self).__init__(config)
        self.kbert = False

        if self.kbert:
            self.albert = KBERT(config)
        else:
            self.albert = AlbertModel(config)            

        self.do_att_merge = not no_att_merge
        self.att_merge = AttentionMerge(
            config.hidden_size, 1024, 0.1) if self.do_att_merge else None

        hidden_layer = 100
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_layer, 1),
        )

        self.n_choices = N_choices

        self.init_weights()
        self.requires_grad = {}

    def freeze_lm(self):
        logger.info('freeze lm layers.')
        for name, p in self.albert.named_parameters():
            self.requires_grad[p] = p.requires_grad
            p.requires_grad = False

    def unfreeze_lm(self):
        logger.info('unfreeze lm layers.')
        for name, p in self.albert.named_parameters():
            p.requires_grad = self.requires_grad[p]

    def forward(self, idx, input_ids, attention_mask, token_type_ids, labels):
        """
        input_ids: [B, N_choices, L]
        labels: [B, ]
        """

        logits = self._forward(idx, input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            predicts = predicts.to(torch.float32).cuda()
            right_num = torch.sum(predicts == labels)
        return loss, right_num, self._to_tensor(idx.size(0), idx.device), logits

    def _forward(self, idx, input_ids, attention_mask, token_type_ids):
        # [B, 2, L] => [B*2, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        
        if self.kbert:
            flat_attention_mask = self.albert.get_attention_mask()
        
        if self.do_att_merge:
            h12 = self.att_merge(outputs[0], flat_attention_mask)
        else:
            h12 = outputs[0][:, 0, :]

        logits = self.scorer(h12).view(-1, self.n_choices)
        logits = F.softmax(logits, dim = 1)

        return logits

    def predict(self, idx, input_ids, attention_mask, token_type_ids):
        """
        return: [B, N_choices]
        """
        return self._forward(idx, input_ids, attention_mask, token_type_ids)

    def _to_tensor(self, it, device): 
        return torch.tensor(it, device=device, dtype=torch.float)