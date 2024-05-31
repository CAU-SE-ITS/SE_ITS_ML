import pinecone

# Pinecone API 키 설정
API_KEY = 'your_pinecone_api_key'  # 여기서 'your_pinecone_api_key'를 실제 API 키로 교체합니다.

# Pinecone 초기화
pinecone.init(api_key=API_KEY, environment='your_pinecone_environment')  # 'your_pinecone_environment'를 실제 환경으로 교체합니다.

# 인덱스 생성
index_name = 'issue-index'
pinecone.create_index(index_name, dimension=768, metric='cosine')
