from transformers import BertTokenizerFast
from utils.predictor import Predictor
from utils.data_loader_maker import DataLoaderMaker
from model import Model

def test(args):
    choice_num = args.choice_num
    batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    drop_last = False
    append_answer_text = 1
    append_descr = 1
    append_tripple = False if args.append_descr == 0 else True
    no_att_merge = False
    model_path = args.model_path
    test_data_path = args.test_data_path
    cache_dir = args.cache_dir
    
    tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

    data_loader_maker = DataLoaderMaker()
    dataloader = data_loader_maker.make(
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

    model = Model.from_pretrained(model_path, cache_dir=cache_dir, no_att_merge=no_att_merge, N_choices = choice_num).cuda()

    predictor = Predictor()
    predictor.predict(Model, model, dataloader)