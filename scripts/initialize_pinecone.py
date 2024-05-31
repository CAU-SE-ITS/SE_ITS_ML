import os
from pinecone import Pinecone, ServerlessSpec

# Pinecone API 키 설정
API_KEY = 'f9da5978-37e0-4044-855f-9fdd04cd7a03'  # 실제 API 키로 교체
os.environ["PINECONE_API_KEY"] = API_KEY

# Pinecone 인스턴스 생성
pc = Pinecone(api_key=API_KEY)

# 인덱스 이름 설정
index_name = 'its'

index = pc.Index(index_name)

print(f"Connected to index '{index_name}'")
