import numpy as np
from transformers import BertTokenizer, BertModel
from pymilvus import Collection
from db import insert_vectors, get_new_issues_and_labels
import torch


# 기존 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 새로운 데이터 로드
new_issues, new_labels = get_new_issues_and_labels()

# 데이터 전처리
inputs = tokenizer(new_issues, truncation=True, padding=True, return_tensors="pt", max_length=512)

# 벡터화
with torch.no_grad():
    outputs = model(**inputs)
    vectors = outputs.last_hidden_state.mean(dim=1).numpy()

# Milvus에 새로운 벡터 추가
insert_vectors(vectors, new_labels)
