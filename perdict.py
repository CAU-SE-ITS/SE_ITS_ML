import pandas as pd
from transformers import BertTokenizer

# 기존 개발자 목록 불러오기
developers = pd.read_csv('developers.csv')

# 새로운 개발자 추가
new_developer = "new_developer_name"
if new_developer not in developers['name'].values:
    developers = developers.append({'name': new_developer}, ignore_index=True)
    developers.to_csv('developers.csv', index=False)

num_developers = len(developers)

# 데이터 전처리 함수
def preprocess_data(data, tokenizer):
    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
    return inputs

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')