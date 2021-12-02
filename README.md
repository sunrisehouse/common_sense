# Common Sense

## Description

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

### Training

```python

python3 src/main.py

```