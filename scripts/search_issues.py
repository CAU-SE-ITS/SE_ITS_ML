import os
from os.path import dirname
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import AutoTokenizer,AutoModel,BertTokenizer, BertModel
from kobert_transformers import get_tokenizer, get_kobert_model
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=API_KEY)

index_name = 'its'

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