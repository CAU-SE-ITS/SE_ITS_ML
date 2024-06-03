import os
from os.path import dirname
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import AutoTokenizer,AutoModel,BertTokenizer, BertModel
import torch
import numpy as np
from dotenv import load_dotenv
from kobert_transformers import get_tokenizer, get_kobert_model

load_dotenv()

API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=API_KEY)

index_name = 'its'

index = pc.Index(index_name)

# KoBERT 모델과 토크나이저 로드
tokenizer = get_tokenizer()
model = get_kobert_model()

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
    {'id': '11', 'title': 'System crash', 'description': 'The system crashes when I try to save a file.', 'category': 'Bug', 'priority': 'High'},
    {'id': '12', 'title': '404 error', 'description': 'Error 404: Page not found.', 'category': 'Bug', 'priority': 'Medium'},
    {'id': '23', 'title': 'Application freeze', 'description': 'The application freezes when opening a large document.', 'category': 'Bug', 'priority': 'High'},
    {'id': '34', 'title': '메뉴 오류', 'description': '메뉴에서 특정 항목을 선택할 때 오류가 발생합니다.', 'category': 'Bug', 'priority': 'Low'},
    {'id': '554', 'title': '로그인 실패', 'description': '사용자가 로그인할 수 없습니다. 비밀번호가 맞는데도 로그인 실패 메시지가 뜹니다.', 'category': 'Bug', 'priority': 'High'},
    {'id': '66', 'title': 'Slow performance', 'description': 'The application is very slow when processing large datasets.', 'category': 'Performance', 'priority': 'Medium'},
    {'id': '77', 'title': 'Translation error', 'description': 'The translation for some terms is incorrect.', 'category': 'Localization', 'priority': 'Low'},
    {'id': '88', 'title': '데이터베이스 연결 실패', 'description': '서버가 데이터베이스에 연결할 수 없습니다. 네트워크 문제로 보입니다.', 'category': 'Bug', 'priority': 'High'},
    {'id': '99', 'title': 'UI issue', 'description': 'The user interface does not scale properly on high-resolution screens.', 'category': 'UI', 'priority': 'Medium'},
    {'id': '010', 'title': '보안 문제', 'description': '사용자 데이터가 암호화되지 않은 채로 전송되고 있습니다.', 'category': 'Security', 'priority': 'Critical'}
]


# 이슈 데이터 삽입
insert_issues(test_issues)

print("Data inserted successfully")