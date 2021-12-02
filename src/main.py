import torch
from transformers import BertTokenizerFast
from utils.trainer import Trainer
from utils.data_loader_maker import DataLoaderMaker
from .model import Model

def train():
    batch_size = 4
    max_seq_length = 128
    drop_last = False
    append_answer_text = 1
    append_descr = 1
    append_tripple = True
    tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

    data_loader_maker = DataLoaderMaker()
    train_dataloader = data_loader_maker.make(
        './data/korqa_train_v1.json',
        tokenizer,
        batch_size,
        drop_last,
        max_seq_length,
        append_answer_text,
        append_descr,
        append_tripple,
        shuffle = False
    )

    devlp_dataloader = data_loader_maker.make(
        './data/korqa_dev_v1.json',
        tokenizer,
        batch_size,
        drop_last,
        max_seq_length,
        append_answer_text,
        append_descr,
        append_tripple,
        shuffle = False
    )

    gpu_ids = None
    if gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        gpu_ids=','.join([str(i) for i in range(n_gpus)])
        print('gpu_ids:', gpu_ids)

    fp16 = 0
    fp16 = True if fp16 == 1 else False
    
    bert_model_dir = 'kykim/albert-kor-base'
    cache_dir = 'cache'
    no_att_merge = False
    print_step = 100
    output_model_dir = './Result/model'
    num_train_epochs = 10
    warmup_proportion = 0.1
    weight_decay = 0.15
    lr = 1e-5
    freeze_lm_epochs = 0

    trainer = Trainer(
        gpu_ids=gpu_ids,
        bert_model_dir=bert_model_dir,
        cache_dir=cache_dir,
        no_att_merge=no_att_merge,
        print_step=print_step,
        output_model_dir=output_model_dir,
        fp16=fp16,
        num_train_epochs=num_train_epochs,
        warmup_proportion=warmup_proportion,
        weight_decay=weight_decay,
        lr=lr,
        freeze_lm_epochs=freeze_lm_epochs,
    )

    trainer.train(
        Model, train_dataloader, devlp_dataloader,
    )

if __name__ == '__main__':
    train()