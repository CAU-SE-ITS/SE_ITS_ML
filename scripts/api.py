import os
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Pinecone API 키 설정
API_KEY = 'f9da5978-37e0-4044-855f-9fdd04cd7a03'  # 실제 API 키로 교체
os.environ["PINECONE_API_KEY"] = API_KEY

pc = Pinecone(api_key=API_KEY)
index_name = 'its'

index = pc.Index(index_name)

# KoBERT 모델과 토크나이저 로드 (Hugging Face를 사용하여 가져오기)
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
model = AutoModel.from_pretrained('monologg/kobert')

def cosine_similarity_to_percentage(cosine_similarity):
    normalized_similarity = (cosine_similarity + 1) / 2
    percentage_similarity = normalized_similarity * 100
    return percentage_similarity

def search_similar_issues(issue, top_k=5):
    text = f"Title: {issue['title']}. Description: {issue['description']}. Category: {issue['category']}. Priority: {issue['priority']}."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0].tolist() 
    
    results = index.query(vector=query_embedding, top_k=top_k)
    
    similar_issues = [
        {"issue_id": int(match.id), "score": cosine_similarity_to_percentage(match.score)}
        for match in results.matches
    ]
    return similar_issues, query_embedding

def insert_issue(issue_id, embedding):
    index.upsert([(issue_id, embedding)])

@app.route('/api/v1/issue/issue_recommend', methods=['POST'])
def issue_recommend():
    data = request.json
    issue_id = data.get('issue_id', '')
    title = data.get('title', '')
    description = data.get('description', '')
    category = data.get('category', '')
    priority = data.get('priority', '')
    
    issue = {
        'issue_id': issue_id,
        'title': title,
        'description': description,
        'category': category,
        'priority': priority
    }
    
    similar_issues, query_embedding = search_similar_issues(issue)
    
    insert_issue(issue_id, query_embedding)
    
    return jsonify(similar_issues)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
