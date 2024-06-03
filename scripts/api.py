import os
from os.path import dirname
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

API_KEY = 'f9da5978-37e0-4044-855f-9fdd04cd7a03'  
os.environ["PINECONE_API_KEY"] = API_KEY

pc = Pinecone(api_key=API_KEY)

index_name = 'its'

index = pc.Index(index_name)

model_dir = '../model'

tokenizer = BertTokenizer.from_pretrained(f'{dirname(__file__)}/../model/')
model = BertModel.from_pretrained(f'{dirname(__file__)}/../model/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def cosine_similarity_to_percentage(cosine_similarity):
    normalized_similarity = (cosine_similarity + 1) / 2
    percentage_similarity = normalized_similarity * 100
    return percentage_similarity

def get_embedding(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # [CLS] 토큰의 임베딩 추출
    return embedding.tolist()

def search_similar_issues(issue, top_k=5):
    text = f"Title: {issue['title']}. Description: {issue['description']}. Category: {issue['category']}. Priority: {issue['priority']}."
    query_embedding = get_embedding(text)
    
    results = index.query(vector=query_embedding, top_k=top_k + 1) 
    
    similar_issues = [
        {"issue_id": int(match.id), "score": cosine_similarity_to_percentage(match.score)}
        for match in results.matches
        if int(match.id) != issue['issue_id']
    ][:top_k]  
    return similar_issues, query_embedding


def insert_issue(issue_id, embedding):
    index.upsert([(str(issue_id), embedding)])

@app.route('/api/v1/issue/issue_recommend', methods=['POST'])
def issue_recommend():
    data = request.json
    issue_id = int(data.get('issue_id')) 
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
