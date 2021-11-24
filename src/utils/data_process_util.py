from rdflib.graph import Graph
from pdb import set_trace
import pprint
import json
import pickle
import random

class Triples():
    def __init__(self):
        self.graph = Graph()

    def readfile(self, filename):
        parse = self.graph.parse(filename, format='nt')
        return parse

class Squad():
    def __init__(self):
        with open('data/squad/ko_wiki_v1_squad.json', 'rt', encoding='UTF8') as f:
            json_data = json.load(f)
        self.squad_data = json_data['data']
        self.squad_titles = []
        self.squad_questions = []
        self.squad_answers = []
        self.squad_contexts = []
        for data in self.squad_data:
            self.squad_titles.append(data['title'])
            self.squad_questions.append(data['paragraphs'][0]['qas'][0]['question'])
            self.squad_answers.append(data['paragraphs'][0]['qas'][0]['answers'][0]["text"])
            self.squad_contexts.append(data['paragraphs'][0]['context'])     

    def extract_new_triples(self, parse):
        self.extracted_triples = []
        for triple in parse:
            head = str(triple[0])
            split = head.split('/')
            text_head = split[-1]
            if text_head in self.squad_titles:
                self.extracted_triples.append(triple)

        with open('data/new_triples', 'wb') as f:
            pickle.dump(self.extracted_triples, f)

    def process_kg(self):
        """ preprocess knowledge graph
            self.Q_choice: {'head entity': [tail entities]}
            self.KG: {'head entity: [[relation, tail], [relation, tail] ...]'}
            Ex.
            >> self.Q_choice['2011_국카스텐_콘서트']
               ['1', '대한민국 서울', '악스코리아', '2011년 7월 9일', '2,500']
            >> self.KG['2011_국카스텐_콘서트']
               [['공연횟수', '1'], ['지역', '대한민국 서울'], ['발생지', '악스코리아'], ['날짜', '2011년 7월 9일'], ['관중수', '2,500']]"""

        with open('data/new_triples', 'rb') as f:
            triples = pickle.load(f)
        self.KG = {}
        self.Q_choice = {}
        for triple in triples:
            tail = str(triple[-1])
            relation = str(triple[1])
            head = str(triple[0])
            split = head.split('/')
            text_head = split[-1]
            split = relation.split('/')
            text_relation = split[-1]
            if text_head in self.KG.keys():
                self.Q_choice[text_head].append(tail)
                self.KG[text_head].append([text_relation, tail])
            else:
                self.Q_choice[text_head] = [tail]
                self.KG[text_head] = []
                self.KG[text_head].append([text_relation, tail])
        
        print(len(self.KG))
        print(len(self.Q_choice))
        f.close()
        with open('data/KG.pickle', 'wb') as f:
            pickle.dump(self.KG, f)
        f.close()
        with open('data/Q_choices.pickle', 'wb') as f:
            pickle.dump(self.Q_choice, f)
        f.close()

    def augment_kg(self):
        with open('data/KG.pickle', 'rb') as f:
            KG = pickle.load(f)
        f.close()
        with open('data/Q_choices.pickle', 'rb') as f:
            Q_choices = pickle.load(f)
        f.close()
        
        self.new_squad_titles = []
        self.new_squad_questions = []
        self.new_squad_answers = []
        self.new_squad_contexts = []

        check_removed_qa = 0
        for i, title in enumerate(self.squad_titles):
            if title not in list(Q_choices.keys()):
                check_removed_qa += 1
                continue
            else:
                self.new_squad_titles.append(self.squad_titles[i])
                self.new_squad_questions.append(self.squad_questions[i])
                self.new_squad_answers.append(self.squad_answers[i])
                self.new_squad_contexts.append(self.squad_contexts[i])
                if self.squad_answers[i] not in Q_choices[title]:
                    Q_choices[title].append(self.squad_answers[i])
                    KG[title].append(['none', self.squad_answers[i]])

        print('Removed QA: ') # 43156
        print(check_removed_qa)
        print('Total QA datset: ') # 25382
        print(len(self.new_squad_titles))
        with open('data/augmented_KG.pickle', 'wb') as f:
            pickle.dump(KG, f)
        f.close()
        with open('data/augmented_Q_choices.pickle', 'wb') as f:
            pickle.dump(Q_choices, f)
        f.close()
        with open('data/squad_titles.pickle', 'wb') as f:
            pickle.dump(self.new_squad_titles, f)
        f.close()
        with open('data/squad_questions.pickle', 'wb') as f:
            pickle.dump(self.new_squad_questions, f)
        f.close()
        with open('data/squad_answers.pickle', 'wb') as f:
            pickle.dump(self.new_squad_answers, f)
        f.close()
        with open('data/squad_contexts.pickle', 'wb') as f:
            pickle.dump(self.new_squad_contexts, f)
        f.close()

class Kor_QA():
    def __init__(self):
        self.all_data = []
        self.kg = None
        self.squad_titles = None
        self.squad_questions = None
        self.squad_answers = None
        self.squad_contexts = None
        with open('data/augmented_KG.pickle', 'rb') as f:
            self.kg = pickle.load(f)
        f.close()
        with open('data/squad_titles.pickle', 'rb') as f:
            self.squad_titles = pickle.load(f)
        f.close()
        with open('data/squad_questions.pickle', 'rb') as f:
            self.squad_questions = pickle.load(f)
        f.close()
        with open('data/squad_answers.pickle', 'rb') as f:
            self.squad_answers = pickle.load(f)
        f.close()
        with open('data/squad_contexts.pickle', 'rb') as f:
            self.squad_contexts = pickle.load(f)
        f.close()

    def same_foramt_json_v1(self):
        num_choices_max = 0
        num_choices_min = 17
        num_choices_avg = 0
        for i, title in enumerate(self.squad_titles):
            new_data = {}
            new_data['initial_id'] = i
            new_data['question'] = {}
            new_data['question']['question_concept'] = title
            new_data['question']['choices'] = []
            new_data['question']['stem'] = self.squad_questions[i]
            new_data['question']['context'] = self.squad_contexts[i]
            triples = self.kg[title]

            num_choices_avg += len(triples)
            if len(triples) > num_choices_max:
                num_choices_max = len(triples)
            if len(triples) < num_choices_min:
                num_choices_min = len(triples)

            tmp_choices = []
            for triple in triples:
                triple_choice = {}
                relation = triple[0]
                tail = triple[1]
                triple_choice['text'] = tail
                triple_choice['triple'] = [[title, relation, tail]]
                if str(tail) == str(self.squad_answers[i]):
                    triple_choice['label'] = 'ans'
                tmp_choices.append(triple_choice)
            random.shuffle(tmp_choices)

            for i, choice in enumerate(tmp_choices):
                if 'label' in choice:
                    if choice['label'] == 'ans':
                        new_data['answerKey'] = i
                choice["label"] = i
                new_data['question']['choices'].append(choice)
            self.all_data.append(new_data)

        num_choices_avg /= len(self.squad_titles)
        print(num_choices_avg) # 16.98
        print(num_choices_max) # 93
        print(num_choices_min) # 2

        train_num = int(len(self.all_data) * 0.7)
        train_data = self.all_data[:train_num]
        dev_data = self.all_data[train_num: train_num + 2500]
        test_data = self.all_data[train_num + 2500 :]

        with open('data/korqa_train_v1.json', 'w', encoding = 'utf-8') as f:
            json.dump(train_data, f)
        f.close()
        with open('data/korqa_dev_v1.json', 'w', encoding = 'utf-8') as f:
            json.dump(dev_data, f)
        f.close()
        with open('data/korqa_test_v1.json', 'w', encoding = 'utf-8') as f:
            json.dump(test_data, f)
        f.close()

#test = Triples()
#parse = test.readfile('data/qa_triples/common_triples@180105.nt')
#squad = Squad()
#squad.extract_new_triples(parse)
#squad.process_kg()
#squad.augment_kg()
kor_qa = Kor_QA()
kor_qa.same_foramt_json_v1()