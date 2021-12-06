from transformers import BertTokenizerFast
from utils.data_loader_maker import DataLoaderMaker
from model import Model
import torch
from utils import clip_batch

def subtask(self, dataloader,model, desc='Eval'):  # 주관식 코드
    total_logits = []
    Answer_list = []#이 부분 검토
    for batch in dataloader:
        batch = clip_batch(batch)
        model.eval()
        batch_labels = batch[4] if self.config.predict_dev else torch.zeros_like(batch[4])
        with torch.no_grad():
            all_ret = model(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch_labels.cuda())
            ret = all_ret[3]
            total_logits.extend(ret.cpu().numpy().tolist())
            Answer_list.extend(batch[4].numpy().tolist())#이 부분 검토
    total_logits = torch.tensor(total_logits)
    Answer = Answer_list[torch.argmax(total_logits)]
    correct_answer = ''  # dataloader에서 타켓값 뽑아주시면됩니다.(정답)
    Question = ''  # 마찬가지로 Question 뽑아주시면됩니다.
    print(f"Question:{Question}\n")
    print(f"correct_answer:{correct_answer}  model answer:{Answer}")
    return Answer


def subjective(args):
    choice_num = args.choice_num
    scorer_hidden = args.scorer_hidden
    version = args.model_version
    batch_size = args.batch_size
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
        shuffle = False
    )

    model = Model.from_pretrained(model_path, cache_dir=cache_dir, no_att_merge=no_att_merge, N_choices = choice_num, scorer_hidden = scorer_hidden, version = version).cuda()

    answer = subtask(dataloader,model)