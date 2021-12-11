# Common Sense

## Description

### Training

```

python3 src/main.py \
    --mission=train \
    --train_data_path=./data/korqa_train_10.json \
    --dev_data_path=./data/korqa_dev_10.json \
    --choice_num=10 \
    --model_version=1 \
    --scorer_hidden=100 \
    --output_model_dir=./Result/output \
    --cache_dir=./Result/cache \

```

### Training From Checkpoint

```

python3 src/main.py \
    --mission=train \
    --train_data_path=./data/korqa_train_10.json \
    --dev_data_path=./data/korqa_dev_10.json \
    --choice_num=10 \
    --model_version=1 \
    --scorer_hidden=100 \
    --model_path=./Result/output \
    --output_model_dir=./Result/output \
    --cache_dir=./Result/cache \

```

### Test

```

python3 src/main.py \
    --mission=test \
    --model_version=1 \
    --scorer_hidden=100 \
    --test_data_path=./data/korqa_test_10.json \
    --choice_num=10 \
    --model_path=./Result/output \
    --cache_dir=./Result/cache \

```

### Subjective task

```

python3 src/main.py \
    --mission=subjective \
    --batch_size=1\
    --model_version=1 \
    --scorer_hidden=100 \
    --test_data_path=./data/subjective_data.json \
    --choice_num=10 \
    --model_path=./Result/output \
    --cache_dir=./Result/cache \

```
### Make Data Loader

```python

from src.utils.data_loader_maker import DataLoaderMaker
from transformers import BertTokenizerFast

batch_size = 4
max_seq_length = 128
drop_last = False
append_answer_text = 1
append_descr = 1
append_tripple = True
tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

data_loader_maker = DataLoaderMaker()
dataloader = data_loader_maker.make(
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

```
