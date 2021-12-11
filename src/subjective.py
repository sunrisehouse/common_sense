from transformers import BertTokenizerFast
from utils.data_loader_maker import DataLoaderMaker
from model import ModelForSub
import torch
from utils import clip_batch
import json
import numpy as np

def subtask(dataloader, model,tdp, desc='Eval'):  # 주관식 코드
    total_logits = []
    Correct_list = []
    f = open(tdp, 'r')
    sub_data = json.load(f)
    for data in sub_data:
         Correct_list.append(data['question']['choices'][int(data['answerKey'])-1]['text'])
    f.close()
    logits_list = []
    k = 5
    check_q =0
    print('[subjective task]')
    for i,batch in enumerate(dataloader):
        batch = clip_batch(batch)
        model.eval()
        batch_labels = batch[4]
        if batch[0][0][-1].item() != check_q:
            total_logits.append(np.array(logits_list).flatten().tolist())
            print(np.shape(total_logits))
            logits_list = []
            check_q += 1
        with torch.no_grad():
            logits = model(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch_labels.cuda())
            logits_list.extend(logits.cpu().numpy().tolist()[0])

    total_logits = torch.tensor(total_logits) # total_logit shape: (total question, 25380)
    predict_label = torch.topk(total_logits,k=k,dim=1)[1]

    Answer = []
    for i, labels in enumerate(predict_label):
        topk_answers = []
        for label in labels:
            label = int(label)
            answer = sub_data[i]['question']['choices'][label]['text']
            topk_answers.append(answer)
            correct_answer = Correct_list[i]
            Question = sub_data[i]['question']['stem']
        print(f"Question{i}:{Question}")
        print(f"correct_answer:{correct_answer}  model answer:{topk_answers}\n")
    return Answer


def subjective(args):
    choice_num = args.choice_num
    scorer_hidden = args.scorer_hidden
    version = args.model_version
    batch_size = 1
    max_seq_length = args.max_seq_length
    drop_last = False
    append_answer_text = args.append_answer_text
    append_descr = args.append_descr
    append_tripple = False if args.append_tripple == 0 else True
    no_att_merge = False
    model_path = args.model_path
    test_data_path = args.test_data_path
    cache_dir = args.cache_dir

    tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

    data_loader_maker = DataLoaderMaker()
    dataloader = data_loader_maker.make(#질문은 동일하고 엔터디만 다르게 들어갈 수 있는 데이터셋 필요
        test_data_path,
        tokenizer,
        batch_size,
        drop_last,
        max_seq_length,
        append_answer_text,
        append_descr,
        append_tripple,
        shuffle = False,
        subjective = True
    )

    model = ModelForSub.from_pretrained(model_path, cache_dir=cache_dir, no_att_merge=no_att_merge, N_choices = choice_num, scorer_hidden = scorer_hidden, version = version).cuda()

    answer = subtask(dataloader=dataloader,model=model, tdp=test_data_path)