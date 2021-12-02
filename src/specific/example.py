# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.feature import Feature
import pdb

class korKGExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, choice6, choice7, choice8, choice9, choice10, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.text6 = choice4
        self.text7 = choice7
        self.text8 = choice8
        self.text9 = choice9
        self.text10 = choice10
        self.label = int(label)
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.text6} | {self.text7} | {self.text8} | {self.text9} | {self.text10} |{self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)
        tokens6 = tokenizer.tokenize(self.text6)
        tokens7 = tokenizer.tokenize(self.text7)
        tokens8 = tokenizer.tokenize(self.text8)
        tokens9 = tokenizer.tokenize(self.text9)
        tokens10 = tokenizer.tokenize(self.text10)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        feature6 = Feature.make_single(self.idx, tokens6, tokenizer, max_seq_length)
        feature7 = Feature.make_single(self.idx, tokens7, tokenizer, max_seq_length)
        feature8 = Feature.make_single(self.idx, tokens8, tokenizer, max_seq_length)
        feature9 = Feature.make_single(self.idx, tokens9, tokenizer, max_seq_length)
        feature10 = Feature.make_single(self.idx, tokens10, tokenizer, max_seq_length)
        return (feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10)
        
        
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
                context = json_obj['question']['context'][:200]
                triples_temp = '{} [SEP] {} [SEP] {}'.format(first_triple, context, following_triple)
            
            text = ' {} [SEP] {} '.format(question_text, triples_temp)
            return text

        text1 = mkinput(question_concept, choices[0])
        text2 = mkinput(question_concept, choices[1])
        text3 = mkinput(question_concept, choices[2])
        text4 = mkinput(question_concept, choices[3])
        text5 = mkinput(question_concept, choices[4])
        text6 = mkinput(question_concept, choices[5])
        text7 = mkinput(question_concept, choices[6])
        text8 = mkinput(question_concept, choices[7])
        text9 = mkinput(question_concept, choices[8])
        text10 = mkinput(question_concept, choices[9])

        label =  int(json_obj['answerKey']) - 1

        return cls(
            json_obj['initial_id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            text6,
            text7,
            text8,
            text9,
            text10,
            label,
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }