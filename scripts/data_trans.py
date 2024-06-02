import pandas as pd
import re
from deep_translator import GoogleTranslator
import time

# CSV 파일 로드
data = pd.read_csv('./data/sev.csv')

# 불필요한 텍스트 제거 함수
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'(Created by [\w\s]+ on [\w\s]+ \d+ \d+)', '', text)  # 'Created by ...' 패턴 제거
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)  # 날짜 패턴 제거 (예: 1998-04-07)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # 이메일 패턴 제거
    return text

# 데이터 전처리
data['Description'] = data['Description'].apply(preprocess_text)

# 번역기 초기화
translator = GoogleTranslator(source='en', target='ko')

# 텍스트를 일정한 길이로 나누는 함수
def split_text(text, max_length=5000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    chunks.append(' '.join(current_chunk))
    return chunks

# 번역 함수
def translate_text(text):
    chunks = split_text(text, max_length=500)
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)  # 잠시 대기 후 다시 시도
            translated_chunk = translator.translate(chunk)  # 재시도
            translated_chunks.append(translated_chunk)
    return ' '.join(translated_chunks)

# 번역할 데이터 나누기
batch_size = 100  # 한 번에 처리할 데이터 양
num_batches = len(data) // batch_size + 1

# 번역 수행
translated_descriptions = []
for i in range(num_batches):
    print(f"Translating batch {i+1}/{num_batches}")
    batch = data['Description'][i*batch_size:(i+1)*batch_size]
    translated_batch = [translate_text(text) for text in batch]
    translated_descriptions.extend(translated_batch)
    time.sleep(5)  # API rate limit을 피하기 위해 잠시 대기

# 번역된 데이터 추가
data['Description_ko'] = translated_descriptions
data['Severity_ko'] = data['Severity']

# 새로운 텍스트 생성
data['combined_text_ko'] = "Description: " + data['Description_ko'] + " Priority: " + data['Severity_ko'].astype(str)

# 번역된 데이터 저장
data.to_csv('/mnt/data/sev_translated.csv', index=False)

print(data[['Description_ko', 'Severity_ko', 'combined_text_ko']].head())
