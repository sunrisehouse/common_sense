# Model Name

## Description
    python main.py --append_answer_text 1 --bert_vocab_dir kykim/albert-kor-base --bert_model_dir kykim/albert-kor-base --batch_size 4 --append_descr 1 --max_seq_length 128 --lr 1e-5 --weight_decay 0.15 --train_file_name ../data/korqa_train_v1.json --output_model_dir ../Result/model --cache_dir cache --devlp_file_name ../data/korqa_dev_v1.json