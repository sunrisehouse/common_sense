# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.feature import Feature
import pdb

#label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

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
