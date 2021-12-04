import torch
from . import clip_batch

class Predictor():
    def  __init__(self):
        print('[predictor]')

    def predict(
        self, model, dataloader,
    ):
        result = []
        idx = []
        labels = []
        predicts = []
    
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