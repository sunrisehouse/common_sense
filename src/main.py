import torch
from transformers import BertTokenizerFast
from utils.trainer import Trainer
from utils.data_loader_maker import DataLoaderMaker
from utils import get_device
from model import Model
import random
import numpy as np

def train(train_dataloader, devlp_dataloader):
    gpu_ids = None
    if gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        gpu_ids=','.join([str(i) for i in range(n_gpus)])
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
    model = Model.from_pretrained(bert_model_dir, cache_dir=cache_dir, no_att_merge=no_att_merge).cuda()
    gpu_ids =  list(map(int, gpu_ids.split(',')))
    multi_gpu = (len(gpu_ids) > 1)
    device = get_device(gpu_ids)

    trainer = Trainer(
        model, multi_gpu, device,
        print_step, output_model_dir, fp16
    )

    optimizer = trainer.make_optimizer(weight_decay, lr)
    scheduler = trainer.make_scheduler(optimizer, warmup_proportion, len(train_dataloader) * num_train_epochs)

    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)

    trainer.train(
        num_train_epochs, train_dataloader, devlp_dataloader, 
        save_last=False, freeze_lm_epochs=freeze_lm_epochs
    )

if __name__ == '__main__':
    seed = 1102
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    train(train_dataloader, devlp_dataloader)