import os
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
import torch

# Pinecone API 키 설정
API_KEY = 'f9da5978-37e0-4044-855f-9fdd04cd7a03'  # 실제 API 키로 교체
os.environ["PINECONE_API_KEY"] = API_KEY

# Pinecone 인스턴스 생성
pc = Pinecone(api_key=API_KEY)

# 인덱스 이름 설정
index_name = 'its'

# 인덱스에 연결
index = pc.Index(index_name)

# BERT 모델과 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 이슈 데이터를 삽입하는 함수
def insert_issues(issues):
    vectors = []
    for issue in issues:
        text = f"Title: {issue['title']}. Description: {issue['description']}. Category: {issue['category']}. Priority: {issue['priority']}."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy().tolist()[0]  # [CLS] 토큰의 임베딩 사용
        vectors.append((issue['id'], embedding))
    index.upsert(vectors)

# 테스트 데이터
test_issues = [
    {'id': '1', 'title': 'System crash', 'description': 'The system crashes when I try to save a file.', 'category': 'Bug', 'priority': 'High'},
    {'id': '2', 'title': '404 error', 'description': 'Error 404: Page not found.', 'category': 'Bug', 'priority': 'Medium'},
    {'id': '3', 'title': 'Application freeze', 'description': 'The application freezes when opening a large document.', 'category': 'Bug', 'priority': 'High'}
]

# 이슈 데이터 삽입
insert_issues(test_issues)

print("Data inserted successfully")