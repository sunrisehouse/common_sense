from rdflib.graph import Graph
from pdb import set_trace
import pprint
import json
import pickle

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
        for data in self.squad_data:
            self.squad_titles.append(data['title'])
            self.squad_questions.append(data['paragraphs'][0]['qas'][0]['question'])
            self.squad_answers.append(data['paragraphs'][0]['qas'][0]['answers'][0]["text"])
        
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

        check_removed_qa = 0
        for i, title in enumerate(self.squad_titles):
            if title not in list(Q_choices.keys()):
                check_removed_qa += 1
                continue
            else:
                self.new_squad_titles.append(self.squad_titles[i])
                self.new_squad_questions.append(self.squad_questions[i])
                self.new_squad_answers.append(self.squad_answers[i])
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


test = Triples()
parse = test.readfile('data/qa_triples/common_triples@180105.nt')
squad = Squad()
#squad.extract_new_triples(parse)
#squad.process_kg()
squad.augment_kg()