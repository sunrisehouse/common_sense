import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--choice_num', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.15)
    parser.add_argument('--max_seq_length', type=int, default=128)

    # Path parameters
    parser.add_argument('--model_path', type=str, default='kykim/albert-kor-base')
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--dev_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--output_model_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    
    # Data parameters
    parser.add_argument('--append_answer_text', type=int, default=1, help='append answer choice to the question.')
    parser.add_argument('--append_descr', type=int, default=1, help='append QA related context description.')
    parser.add_argument('--append_tripple', type=bool, default=1, help='appending triples so we use external Knowledge Graph.')

    # Other parameters
    parser.add_argument('--print_step', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=1102)
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--predict_dev', action='store_true', help='predict results on dev.')
    parser.add_argument('--fp16', type=int, default=0)


    args = parser.parse_args()

    return args
