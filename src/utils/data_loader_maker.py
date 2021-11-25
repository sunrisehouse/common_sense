import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import json

class korKGExample:
    def __init__(self, idx, choices, label = -1):
        self.idx = idx
        self.texts = choices
        self.label = int(label)
   
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens = []
        for i, text in enumerate(self.texts):
            tmp_tokens = tokenizer.tokenize(self.texts[i])
            tokens.append(tmp_tokens)
        
        features = []
        for i, token in enumerate(tokens):
            tmp_feature = Feature.make_single(self.idx, tokens[i], tokenizer, max_seq_length)
            
            features.append(tmp_feature)

        return features

    @classmethod
    def load_from_json(cls, json_obj, append_answer_text=False, append_descr=0, append_triple=True):
        choices = json_obj['question']['choices']
        question_concept = json_obj['question']['question_concept']
        def mkinput(question_concept, choice):
            if choice['triple'] and append_triple:
                triples = ' [SEP] '.join([' '.join(trip) for trip in choice['triple']])
                first_triple = ' '.join(choice['triple'][0])
                following_triple = ' [SEP] '.join([' '.join(trip) for trip in choice['triple'][1:]]) if len(choice['triple']) > 1 else None
                triples_temp = triples
            else:
                triples_temp = question_concept + ' [SEP] ' + choice['text']
                following_triple = None
            if append_answer_text:
                question_text = '{} {}'.format(json_obj['question']['stem'], choice['text'])
            else:
                question_text = json_obj['question']['stem']
            if append_descr == 1:
                context = json_obj['question']['context'][:100]
                triples_temp = '{} [SEP] {} [SEP] {}'.format(first_triple, context, following_triple)
            
            text = ' {} [SEP] {} '.format(question_text, triples_temp)
            return text
        
        texts = []
        for i, choice in enumerate(choices):
            cmd = 'texts.append(mkinput(question_concept, choices[%d]))'%(i)
            exec(cmd)

        try:
            label =  int(json_obj['answerKey'])
        except:
            label = -1
        return cls(
            json_obj['initial_id'],
            texts,
            label,
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Texts': self.texts,
            'Label': self.label
        }


class Feature:
    def __init__(self, idx, input_ids, input_mask, segment_ids):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

    @classmethod
    def make(cls, idx, tokens1, tokens2, tokenizer, max_seq_length):

        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2
        tokens = tokens[:max_seq_length-1]
        tokens = tokens + ['[SEP]']

        input_mask = [1] * len(tokens)
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens) - len(tokens1) - 2)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        if len(segment_ids) > max_seq_length:
            segment_ids = segment_ids[:max_seq_length]
        
        assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length
        return cls(idx, input_ids, input_mask, segment_ids)

    @classmethod
    def make_single(cls, idx, tokens1, tokenizer, max_seq_length):
        tokens = ['[CLS]'] + tokens1
        tokens = tokens[:max_seq_length-1]
        tokens = tokens + ['[SEP]']

        input_mask = [1] * len(tokens)
        segment_ids = [1] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return cls(idx, input_ids, input_mask, segment_ids)
        
    @classmethod
    def make_combined(cls, idx, tokens11, tokens12, tokens2, tokenizer, max_seq_length):

        tokens = ['[CLS]'] + tokens11 + ['[SEP]'] + tokens12
        sent_length = len(tokens)
        tokens = tokens + ['[SEP]'] + tokens2
        tokens = tokens[:max_seq_length-1]
        tokens = tokens + ['[SEP]']

        input_mask = [1] * len(tokens)
        segment_ids = [0] * sent_length

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += [1]  * (max_seq_length - sent_length)
        segment_ids = segment_ids[:max_seq_length]
        assert len(segment_ids) == max_seq_length

        return cls(idx, input_ids, input_mask, segment_ids)
        
class DataLoaderMaker:
    def __init__(self):
        print("data loader maker")
    
    def make(self, data_file_name, tokenizer, batch_size, drop_last, max_seq_length, append_answer_text, append_descr, append_triple, shuffle=True):
        print("1111")
        examples = self._load_data_korKG(
            data_file_name,
            append_answer_text=append_answer_text, 
            append_descr=append_descr,
            append_triple=append_triple
        )
        print('examples: ', examples)

        F = []
        L = []

        for i, example in enumerate(examples):
            features, la = example.fl(tokenizer, max_seq_length)

            one_hot = np.zeros(shape = (len(features), ), dtype=np.int8)
            one_hot = one_hot.tolist()
            one_hot[int(la)-1] = 1

            F.extend(features)
            L.extend(one_hot)

        return self.convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
    
    def convert_to_tensor(self, data, batch_size, drop_last, shuffle):
        tensors = []

        for item in data:
            # item: (F, L)
            # item[0] = F: [utils.feature.Feature object, utils.feature.Feature object, ...]
            # item[1] = L: tensor([0, 0, 1, 0, 0 ...])  --> each one-hot label
            if type(item[0]) is Feature:
                _tensors = self._convert_feature_to_tensor(item)
                tensors.extend(_tensors)

            elif type(item[0]) is tuple:
                if type(item[0][0]) is Feature:
                    _tensors = self._convert_multi_feature_to_tensor(item)
                    tensors.extend(_tensors)

            elif type(item[0]) is int:
                _tensor = torch.tensor(item, dtype=torch.long)
                tensors.append(_tensor)

            elif type(item[0]) is list:
                if type(item[0][0]) is int:
                    _tensor = torch.tensor(item, dtype=torch.long)
                elif type(item[0][0]) is float:
                    _tensor = torch.tensor(item, dtype=torch.float)

                tensors.append(_tensor)

            else:
                raise Exception(str(type(item[0])))

        # Total Dataset num: 303284

        # tensors[0].shape : torch.Size([303284])  --> Question type: 관련 question id
        # tensors[1].shape: torch.Size([303284, 128]) --> Input ids
        # tensors[2].shape: torch.Size([303284, 128]) --> Input mask
        # tensors[3].shape: torch.Size([303284, 128]) --> Segment ids
        # tensors[4].shape: torch.Size([303284]) --> label: 0(not answer), 1(answer)

        dataset = TensorDataset(*tensors)

        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=batch_size, drop_last=drop_last)
        return dataloader
    
    def _convert_feature_to_tensor(self, features):
        """
        features: [f, f, f, ...]
        """
        all_idx = torch.tensor([f.idx for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        return all_idx, all_input_ids, all_input_mask, all_segment_ids


    def _convert_multi_feature_to_tensor(self, features):
        """
        features: [(f1, f2, f3), ...]
        """
        all_idx = torch.tensor([[f.idx for f in fs] for fs in features], dtype=torch.long)
        all_input_ids = torch.tensor([[f.input_ids for f in fs] for fs in features], dtype=torch.long)
        all_input_mask = torch.tensor([[f.input_mask for f in fs] for fs in features], dtype=torch.long)
        all_segment_ids = torch.tensor([[f.segment_ids for f in fs] for fs in features], dtype=torch.long)
        return all_idx, all_input_ids, all_input_mask, all_segment_ids
    
    def _load_data_korKG(self, file_name, append_answer_text=False, append_descr=False, append_triple=True):
        examples = []
        
        for json_obj in self._load_json(file_name):
            example = korKGExample.load_from_json(json_obj, append_answer_text, append_descr, append_triple)
            examples.append(example)

        return examples
    
    def _load_json(file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            return json.load(f)