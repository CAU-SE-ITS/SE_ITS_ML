from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Connect to Milvus
def connect_milvus():
    connections.connect("default", host='localhost', port='19530')

# Initialize Milvus collection
def initialize_collection():
    fields = [
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="description_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    ]
    schema = CollectionSchema(fields, "issue_collection")
    collection = Collection(name="issue_collection", schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {}})
    collection.load()
    return collection

# Concatenate fields into a single string
def concatenate_fields(title, description, category, priority):
    return f"Title: {title}. Description: {description}. Category: {category}. Priority: {priority}."

# Insert embeddings into Milvus
def insert_embeddings(collection, issues):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    texts = [concatenate_fields(issue['title'], issue['description'], issue['category'], issue['priority']) for issue in issues]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embeddings
    
    collection.insert([embeddings.tolist()])

# Search similar issues in Milvus
def search_similar_issues(collection, issue, top_k=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    text = concatenate_fields(issue['title'], issue['description'], issue['category'], issue['priority'])
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding.tolist()], "embedding", search_params, limit=top_k)
    
    similar_issues = [(result.id, 100 - result.distance) for result in results[0]]  # Calculate similarity percentage
    return similar_issues

# Disconnect from Milvus
def disconnect_milvus():
    connections.disconnect()
