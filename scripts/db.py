from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

# Milvus 연결 설정
def connect_milvus():
    connections.connect("default", host="localhost", port="19530")

# 컬렉션 생성
def create_collection():
    connect_milvus()
    if not utility.has_collection("issue_collection"):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="label", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, "issue collection")
        collection = Collection(name="issue_collection", schema=schema)

# 벡터 삽입
def insert_vectors(vectors, labels):
    connect_milvus()
    collection = Collection("issue_collection")
    entities = [
        vectors.tolist(),
        labels
    ]
    collection.insert(entities)
    collection.load()

# 이슈 및 레이블 가져오기
def get_issues_and_labels():
    # 데이터베이스에서 이슈와 레이블을 가져오는 코드
    pass

# 새로운 이슈 및 레이블 가져오기
def get_new_issues_and_labels():
    # 데이터베이스에서 새로운 이슈와 레이블을 가져오는 코드
    pass
