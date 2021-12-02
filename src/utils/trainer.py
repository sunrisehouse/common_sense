from tqdm.autonotebook import tqdm
import torch
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
import os
from torch.utils.tensorboard import SummaryWriter
from . import get_device, AvgVar, Vn

import logging;

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clip_batch(batch):
    """
    clip batch based on max length
    """
    # print("batch size is {}".format(len(batch[0])))
    idx, input_ids, attention_mask, token_type_ids, labels = batch
    # [batch_size, 2, L]
    batch_size = input_ids.size(0)
    while True:
        end_flag = False
        for i in range(batch_size):
            #if input_ids[i, 0, -1] != 0:
            if input_ids[i, -1] != 0:
                end_flag = True
            # if input_ids[i, 1, -1] != 0:
            #     end_flag = True
            #
        if end_flag:
            break
        else:
            input_ids = input_ids[:, :-1]
            # input_ids = input_ids[:, :, :-1]

    max_seq_length = input_ids.size(1)#2
    attention_mask = attention_mask[:, :max_seq_length]
    token_type_ids = token_type_ids[:, :max_seq_length]
    
    return idx, input_ids, attention_mask, token_type_ids, labels

def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class Trainer:
    def __init__(
        self,
        gpu_ids,
        bert_model_dir,
        cache_dir,
        no_att_merge,
        print_step,
        output_model_dir,
        fp16,
        num_train_epochs,
        warmup_proportion,
        weight_decay,
        lr,
        freeze_lm_epochs,
    ):
        self.gpu_ids = gpu_ids
        self.bert_model_dir = bert_model_dir
        self.cache_dir = cache_dir
        self.no_att_merge = no_att_merge
        self.print_step = print_step
        self.output_model_dir = output_model_dir
        self.fp16 = fp16
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.lr = lr
        self.freeze_lm_epochs = freeze_lm_epochs

    def train(
        self, ModelClass, model, multi_gpu, device,
        train_dataloader, devlp_dataloader,
    ):
        
        gpu_ids =  list(map(int, self.gpu_ids.split(',')))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)
        print('init_model', self.bert_model_dir)
        model = ModelClass.from_pretrained(self.bert_model_dir, cache_dir=self.cache_dir, no_att_merge=self.no_att_merge).cuda()
        print(model)
        
        trainer = ModelTrainer(
            model, multi_gpu, device,
            self.print_step, self.output_model_dir, self.fp16)

        t_total = len(train_dataloader) * self.num_train_epochs
        warmup_proportion = self.warmup_proportion

        optimizer = trainer.make_optimizer(self.weight_decay, self.lr)
        scheduler = trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        trainer.set_optimizer(optimizer)
        trainer.set_scheduler(scheduler)

        trainer.train(
            self.num_train_epochs, train_dataloader, devlp_dataloader, 
            save_last=False, freeze_lm_epochs=self.freeze_lm_epochs)
    
    def trial(self, dataloader, desc='Eval'):
        result = []
        idx = []
        labels = []
        predicts = []

        for batch in dataloader:
            batch = clip_batch(batch)
            self.model.eval()
            batch_labels = batch[4] if self.config.predict_dev else torch.zeros_like(batch[4])
            with torch.no_grad():

                all_ret = self.model(batch[0],batch[1],batch[2],batch[3],batch_labels)
                # all_ret = self.model(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch_labels.cuda())
                ret = all_ret[3]
                idx.extend(batch[0].cpu().numpy().tolist())
                result.extend(ret.cpu().numpy().tolist())
                labels.extend(batch[4].numpy().tolist())
                predicts.extend(torch.argmax(ret, dim=1).cpu().numpy().tolist())
        return idx, result, labels, predicts

class BaseTrainer:
    def __init__(self, model, multi_gpu, device, print_step, output_model_dir, vn):
        self.model = model.to(device)
        self.device = device
        self.multi_gpu = multi_gpu
        self.print_step = print_step
        self.output_model_dir = output_model_dir

        self.vn = vn
        self.train_record = Vn(vn)

    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            self.model = model
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train(self, epoch_num, train_dataloader, dev_dataloader,
              save_last=True, freeze_lm_epochs=0):

        best_dev_loss = float('inf')
        best_dev_acc = 0
        self.global_step = 0
        self.train_record.init()
        self.model.zero_grad()
        if freeze_lm_epochs > 0:
            if self.multi_gpu:
                self.model.module.freeze_lm()
            else:
                self.model.freeze_lm()

        for epoch in range(int(epoch_num)):
            if epoch == freeze_lm_epochs and freeze_lm_epochs > 0:
                if self.multi_gpu:
                    self.model.module.unfreeze_lm()
                else:
                    self.model.unfreeze_lm()
            print(f'---- Epoch: {epoch+1:02} ----')
            for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
                self.model.train()
                self._step(batch)

                if self.global_step % self.print_step == 0:

                    dev_record = self.evaluate(dev_dataloader)
                    self.model.zero_grad()

                    self._report(self.train_record, dev_record)
                    current_acc = dev_record.list()[1]
                    if current_acc > best_dev_acc:
                        best_dev_acc = current_acc
                        self.save_model()

                    self.train_record.init()

        dev_record = self.evaluate(dev_dataloader)
        self._report(self.train_record, dev_record)

        if save_last:
            self.save_model()

    def _forward(self, batch, record):
        batch = tuple(t.to(self.device) for t in batch)
        loss, acc = self.model(*batch)
        loss, acc = self._mean_sum((loss, acc))
        record.inc([loss.item(), acc.item()])
        return loss

    def _mean(self, tuples):
        if self.multi_gpu:
            return tuple(v.mean() for v in tuples)
        return tuples

    def _mean_sum(self, tuples):
        """
        mean if float, sum if int
        """
        if self.multi_gpu:
            return tuple(v.mean() if v.is_floating_point() else v.sum() for v in tuples)
        return tuples

    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1

    def evaluate(self, dataloader, desc='Eval'):
        record = Vn(self.vn)
        print('model eval')
        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                self._forward(batch, record)

        return record

    def _report(self, train_record, devlp_record):
        tloss, tacc = train_record.avg()
        dloss, dacc = devlp_record.avg()
        print("\t\tTrain loss %.4f acc %.4f | Dev loss %.4f acc %.4f" % (
                tloss, tacc, dloss, dacc))

    def make_optimizer(self, weight_decay, lr):
        params = list(self.model.named_parameters())

        no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        def _no_decay(n):
            return any(nd in n for nd in no_decay_keywords)

        parameters = [
            {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not _no_decay(n)],
             'weight_decay': weight_decay}
        ]

        optimizer = AdamW(parameters, lr=lr, eps=1e-8)
        return optimizer

    def make_scheduler(self, optimizer, warmup_proportion, t_total):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)

    def save_model(self):
        mkdir_if_notexist(self.output_model_dir)
        logger.info('saving model {}'.format(self.output_model_dir))
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.output_model_dir)

class ModelTrainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, print_step,
                 output_model_dir, fp16):

        super(ModelTrainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, vn=3)
        self.fp16 = fp16
        self.multi_gpu = multi_gpu
        self.tb_step = 0
        self.tb_writer = SummaryWriter(log_dir=output_model_dir)

        print("fp16 is {}".format(fp16))
        
    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1) 
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1
        
    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            
            self.model = model
        self.optimizer = optimizer

    def _forward(self, batch, record):
        batch = clip_batch(batch)
        batch = tuple(t.to(self.device) for t in batch)
        all_result = self.model(*batch)
        result = all_result[:3]
        result = tuple([result[0].mean(), result[1].sum(), result[2].sum()])
        record.inc([it.item() for it in result])
        return result[0]

    def _report(self, train_record, devlp_record):
        # record: loss, right_num, all_num
        train_loss = train_record[0].avg()
        devlp_loss = devlp_record[0].avg()

        trn, tan = train_record.list()[1:]
        drn, dan = devlp_record.list()[1:]

        logger.info(f'\n____Train: loss {train_loss:.4f} {int(trn)}/{int(tan)} = {int(trn)/int(tan):.4f} |'
              f' Devlp: loss {devlp_loss:.4f} {int(drn)}/{int(dan)} = {int(drn)/int(dan):.4f}')
        self.tb_writer.add_scalar("train_accuracy", int(trn)/int(tan), global_step=self.tb_step)
        self.tb_writer.add_scalar("dev_accuracy", int(drn)/int(dan), global_step=self.tb_step)
        self.tb_writer.add_scalar("train_loss", train_loss, global_step=self.tb_step)
        self.tb_writer.add_scalar("dev_loss", devlp_loss, global_step=self.tb_step)
        self.tb_step += 1
