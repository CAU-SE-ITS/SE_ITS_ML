import os
from os.path import dirname
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import AutoTokenizer,AutoModel,BertTokenizer, BertModel
import torch
import numpy as np
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=API_KEY)

index_name = 'its'

index = pc.Index(index_name)

#tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
#model = AutoModel.from_pretrained('monologg/kobert')

tokenizer = AutoTokenizer.from_pretrained(f'{dirname(__file__)}/../model/')
model = AutoModel.from_pretrained(f'{dirname(__file__)}/../model/')

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def rescale_scores(scores):
    scores = np.array(scores)
    if len(scores) == 1:
        return [50.0]
    min_score = np.min(scores)
    max_score = np.max(scores)
    if min_score == max_score:
        return [50.0] * len(scores)  
    scaled_scores = (scores - min_score) / (max_score - min_score) * 100
    return scaled_scores.tolist()


def get_embedding(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        normalized_embedding = embedding / np.linalg.norm(embedding)  # 벡터 정규화
    return normalized_embedding.tolist()

def search_similar_issues(issue, top_k=5):
    text = f"{issue['title']}.{issue['description']}{issue['category']}.{issue['priority']}."
    query_embedding = get_embedding(text)
    
    try:
        results = index.query(vector=query_embedding, top_k=top_k + 1)
    except Exception as e:
        return [], query_embedding

    if not results.matches:
        return [], query_embedding
    
    absolute_scores = [match.score for match in results.matches if int(match.id) != issue['issue_id']]
    if not absolute_scores:
        return [], query_embedding
    
    relative_scores = rescale_scores(absolute_scores)
    
    combined_scores = [(abs_score * 50) + (rel_score * 0.5) for abs_score, rel_score in zip(absolute_scores, relative_scores)]
    
    similar_issues = [
        {"issue_id": int(match.id), "score": combined_score}
        for match, combined_score in zip(results.matches, combined_scores)
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
    
    print(issue)
    similar_issues, query_embedding = search_similar_issues(issue)
    
    insert_issue(issue_id, query_embedding)
    
    return jsonify(similar_issues)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug = True)
