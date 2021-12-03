import torch
from transformers import BertTokenizerFast
from utils.trainer import Trainer, clip_batch
from utils.data_loader_maker import DataLoaderMaker
from utils import get_device
from model import Model
from option import get_args
import random
import numpy as np

def train(train_dataloader, devlp_dataloader, args):
    gpu_ids = None
    if gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        gpu_ids=','.join([str(i) for i in range(n_gpus)])
    fp16 = 0
    fp16 = True if fp16 == 1 else False
    bert_model_dir = 'kykim/albert-kor-base'
    cache_dir = 'cache'
    no_att_merge = False
    print_step = args.print_step
    output_model_dir = args.predict_dir
    num_train_epochs = args.num_train_epochs
    warmup_proportion = args.warmup_proportion
    weight_decay = args.weight_decay
    lr = args.lr
    freeze_lm_epochs = 0
    model = Model.from_pretrained(bert_model_dir, cache_dir=cache_dir, no_att_merge=no_att_merge, N_choices = args.choice_num).cuda()
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

def trial(self, dataloader, args, desc='Eval'):
    result = []
    idx = []
    labels = []
    predicts = []
    cache_dir = 'cache'
    output_model_dir = args.predict_dir
    model = Model.from_pretrained(output_model_dir, cache_dir=cache_dir, no_att_merge=no_att_merge, N_choices = args.choice_num).cuda()

    for batch in dataloader:
        batch = clip_batch(batch)
        model.eval()
        batch_labels = batch[4]
        with torch.no_grad():
            all_ret = model(batch[0],batch[1],batch[2],batch[3],batch_labels)
            #all_ret = self.model(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch_labels.cuda())
            ret = all_ret[3]
            idx.extend(batch[0].cpu().numpy().tolist())
            result.extend(ret.cpu().numpy().tolist())
            labels.extend(batch[4].numpy().tolist())
            predicts.extend(torch.argmax(ret, dim=1).cpu().numpy().tolist())
    return idx, result, labels, predicts

if __name__ == '__main__':
    args = get_args()

    #### Random Seed ####
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #### Training Setting ####
    choice_num = args.choice_num
    batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    drop_last = False
    append_answer_text = args.append_answer_text
    append_descr = args.append_descr
    append_tripple = args.append_tripple

    tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

    if args.mission == "train":
        #### DataLoader ####
        train_file_name = '../data/korqa_train_' + str(choice_num) + '.json'
        dev_file_name = '../data/korqa_dev_' + str(choice_num) + '.json'

        data_loader_maker = DataLoaderMaker()
        train_dataloader = data_loader_maker.make(
            train_file_name,
            tokenizer,
            batch_size,
            drop_last,
            max_seq_length,
            append_answer_text,
            append_descr,
            append_tripple,
            shuffle = True
        )

        devlp_dataloader = data_loader_maker.make(
            dev_file_name,
            tokenizer,
            batch_size,
            drop_last,
            max_seq_length,
            append_answer_text,
            append_descr,
            append_tripple,
            shuffle = False
        )

        train(train_dataloader, devlp_dataloader, args)

    elif args.mission == 'output':
        test_file_name = '../data/korqa_test_' + str(choice_num) + '.json'

        data_loader_maker = DataLoaderMaker()
        test_dataloader = data_loader_maker.make(
            test_file_name,
            tokenizer,
            batch_size,
            drop_last,
            max_seq_length,
            append_answer_text,
            append_descr,
            append_tripple,
            shuffle = False
        )

        idx, result, label, predict = trial(test_dataloader, args)