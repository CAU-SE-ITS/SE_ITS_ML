import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import Collection, utility
from db import insert_vectors, create_collection

# 데이터 로드
issues, labels = get_issues_and_labels()

# 데이터 전처리
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(issues, truncation=True, padding=True, return_tensors="pt", max_length=512)

# 벡터화
with torch.no_grad():
    outputs = model(**inputs)
    vectors = outputs.last_hidden_state.mean(dim=1).numpy()

# Milvus에 벡터 저장
create_collection()
insert_vectors(vectors, labels)
