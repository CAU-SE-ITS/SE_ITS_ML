import pandas as pd
import re
from datasets import Dataset, load_metric
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import numpy as np

# CSV 파일 로드
data = pd.read_csv('./data/sev.csv')

# 불필요한 텍스트 제거 함수
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'(Created by [\w\s]+ on [\w\s]+ \d+ \d+)', '', text)  # 'Created by ...' 패턴 제거
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)  # 날짜 패턴 제거 (예: 1998-04-07)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # 이메일 패턴 제거
    return text

# 데이터 전처리
data['description'] = data['description'].apply(preprocess_text)
data = data.rename(columns={'severity': 'priority'})  # severity를 priority로 변경

# 데이터셋 만들기
dataset = Dataset.from_pandas(data[['description', 'priority']])

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['description'], padding="max_length", truncation=True, max_length=512)

# 데이터 토큰화
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 데이터셋 포맷 변환
tokenized_datasets = tokenized_datasets.remove_columns(["description"])
tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")
tokenized_datasets.set_format("torch")

# 데이터셋 분리
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 데이터 콜레이터
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델 로드
model = BertModel.from_pretrained('bert-base-uncased')

# 훈련 인수 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.mse_loss(outputs.last_hidden_state[:, 0, :], labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 모델 훈련
trainer.train()

# 모델 저장
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
