# Common Sense

## Description

### Data Source

데이터셋명: 일반상식

데이터 분야: 음성 / 자연어

데이터 유형: 텍스트

구축 년도: 2017년

최종 수정 일자: 2018.01.02

소개: 한국어 위키백과 내 주요 문서 15만 개에 포함된 지식을 추출하여 객체(entity), 속성(attribute), 값(value)을 갖는 트리플 형식의 데이터 75만 개를 구축한 지식베이스 제공.

링크: https://aihub.or.kr/aidata/84


해당 프로젝트에서는 AI HUB에 공개된 ‘일반상식’ 텍스트 데이터를 사용하여 일반상식 QA를 수행했다. 공개된 ‘일반상식’ 데이터셋은 2가지 종류의 데이터로 구성되었다.
하나는 일반상식 지식베이스 그래프이다. 일반 상식 지식베이스 그래프는 한국어 위키백과 문서 15만 개 정보를 기반으로 한 entity, attribute, value 형태의 트리플 데이터이다. 이때 entity들은 위키피디아 표제어의 집합이고, attribute은 각 표제어가 가질 수 있는 정보의 속성, value는 위키피디아 표제어의 속성에 대응하는 값을 나타낸다.  한 트리플의 예시는 다음과 같다: ['교황', '관저', '사도 궁전']. 제공된 일반상식 그래프 트리플 75만개 중 이후 소개할 QA 데이터셋과 관련있는 entity들로 구성된 subgraph(트리플 27000여 개)를 추출하여 우리 프로젝트에 사용했다.
두 번째 데이터는 일반상식 QA 데이터셋이다. 일반상식 QA 데이터셋은 일반상식 관련 (질문-답) 쌍과 해당 답을 추론할 수 있는 정보를 담고 있는 문장(context)으로 구성되었다. 
예를 들어, 질문 "일반성면의 면적이 얼마야"가 주어졌을 때 그에 대한 답인 "19.41 km²과 context 정보 “일반성면은 동부 5개 면의 교통, 문화, 교육, 상업의 중심지로서 .... 면적은 19.41 km²로 …”가 데이터셋에 포함되었다. 또한 일반상식 그래프를 기반으로 만들어진 다른 일반상식 QA데이터셋(CommonsenseQA[1], OpenBookQA[2])과 달리 AI HUB에 주어진 QA 데이터셋은 일반상식 그래프에 정답이 존재하지 않는 질문들이 존재한다. 정답 value가 일반 상식 그래프에 없는 경우, 임의적으로 모델 input으로 들어갈 트리플 형태를 [entity, ‘[SEP]’, value] 으로 변경하여 relation 정보가 존재하지 않음을 나타냈다.

### Data

./data/korqa_train_5.json
./data/korqa_train_10.json
./data/korqa_train_15.json
./data/korqa_train_20.json
./data/korqa_dev_5.json
./data/korqa_dev_10.json
./data/korqa_dev_15.json
./data/korqa_dev_20.json
./data/korqa_test_5.json
./data/korqa_test_10.json
./data/korqa_test_15.json
./data/korqa_test_20.json

### Hyperparameter

- mission(default=train): train, test, subjective 중 수행할 실험을 결정하는 hyperparameter이다.
- choice_num(default=5): 객관식문항 몇개중에 답을 고를것인지를 정하는 hyperparameter로 우리 실험은 10, 20을 중점적으로 진행했다.
- scorer_hidden(default=100): scorer는 마지막 fully connected layer를 의미하며 여기서 사용된 hidden layer의 size를 결정하는 hyperparameter이다.
- model_version(default=1): model은 총 3가지가 있으며 각 모델은 hidden layer의 갯수가 다르다. 이 hyperparameter는 model의 version을 결정한다.
- lr(default=1e-5): learning rate를 조절하는 hyperparameter이다.
- batch_size(default=4): batch_size를 조절하는 hyperparameter이다.
- num_train_epochs(default=10): epoch수 를 조절하는 hyperparameter이다.
- weight_decay(default=0.15): weight_decay를 조절하는 hyperparameter이다.
- max_seq_length(default=128): 최대 sequence 길이를 정하는 hyperparameter이다.


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
