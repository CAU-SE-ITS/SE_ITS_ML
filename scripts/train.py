import pandas as pd
from transformers import BertTokenizer, BertModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import torch

# Load the dataset
file_path = '/mnt/data/sev.csv'
df = pd.read_csv(file_path)

# Select relevant columns
descriptions = df['Description'].tolist()
severities = df['Severity'].tolist()

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize the data
inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embeddings

# Normalize severity values and append to embeddings
severity_map = {severity: idx for idx, severity in enumerate(set(severities))}
normalized_severities = [severity_map[severity] for severity in severities]
normalized_severities = torch.tensor(normalized_severities).unsqueeze(1).numpy()  # Reshape for concatenation

# Concatenate embeddings with severity
final_embeddings = np.hstack((embeddings, normalized_severities))

# Connect to Milvus
connections.connect("default", host='localhost', port='19530')

# Define schema for Milvus collection
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=769),
    FieldSchema(name="description_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
]
schema = CollectionSchema(fields, "IssueEmbeddings")

# Create collection
collection = Collection(name="issue_collection", schema=schema)
collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {}})

# Insert embeddings into Milvus
collection.insert([final_embeddings.tolist()])
collection.load()

# Save model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Disconnect from Milvus
connections.disconnect()
