import os
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
import torch
from kobert_transformers import get_tokenizer, get_kobert_model

# Pinecone API 키 설정
API_KEY = 'f9da5978-37e0-4044-855f-9fdd04cd7a03'  # 실제 API 키로 교체
os.environ["PINECONE_API_KEY"] = API_KEY

# Pinecone 인스턴스 생성
pc = Pinecone(api_key=API_KEY)

# 인덱스 이름 설정
index_name = 'its'

# 인덱스에 연결
index = pc.Index(index_name)

# KoBERT 모델과 토크나이저 로드
tokenizer = get_tokenizer()
model = get_kobert_model()

# 유사한 이슈를 검색하는 함수
def search_similar_issues(issue, top_k=5):
    text = f"Title: {issue['title']}. Description: {issue['description']}. Category: {issue['category']}. Priority: {issue['priority']}."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy().tolist()[0]  # [CLS] 토큰의 임베딩 사용
    
    results = index.query(vector=query_embedding, top_k=top_k)
    return results

# 새로운 이슈
new_issue = {'title': 'Document save crash', 'description': 'The application crashes when saving a document.', 'category': 'Bug', 'priority': 'High'}

# 유사한 이슈 검색
similar_issues = search_similar_issues(new_issue)
print("Similar issues:", similar_issues)