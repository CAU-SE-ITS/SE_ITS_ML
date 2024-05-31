from db import connect_milvus, initialize_collection, insert_embeddings, search_similar_issues, disconnect_milvus

def test_milvus():
    # 연결 설정
    connect_milvus()
    
    # 컬렉션 초기화
    collection = initialize_collection()
    
    # 테스트 데이터
    test_issues = [
        {
            'title': 'System crash',
            'description': 'The system crashes when I try to save a file.',
            'category': 'Bug',
            'priority': 'High'
        },
        {
            'title': '404 error',
            'description': 'Error 404: Page not found.',
            'category': 'Bug',
            'priority': 'Medium'
        },
        {
            'title': 'Application freeze',
            'description': 'The application freezes when opening a large document.',
            'category': 'Bug',
            'priority': 'High'
        }
    ]
    
    # 데이터 삽입
    insert_embeddings(collection, test_issues)
    
    # 유사한 이슈 검색
    new_issue = {
        'title': 'Document save crash',
        'description': 'The application crashes when saving a document.',
        'category': 'Bug',
        'priority': 'High'
    }
    similar_issues = search_similar_issues(collection, new_issue)
    
    print("Similar issues:", similar_issues)
    
    # 연결 해제
    disconnect_milvus()

if __name__ == "__main__":
    test_milvus()
