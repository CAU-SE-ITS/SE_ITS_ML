import pandas as pd
import re
from deep_translator import GoogleTranslator
import time

data = pd.read_csv('./data/sev_test.csv')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'(Created by [\w\s]+ on [\w\s]+ \d+ \d+)', '', text)  # 'Created by ...' 패턴 제거
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)  # 날짜 패턴 제거 (예: 1998-04-07)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # 이메일 패턴 제거
    return text

data['Description'] = data['Description'].apply(preprocess_text)

translator = GoogleTranslator(source='en', target='ko')

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
    chunks = split_text(text, max_length=1000)
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)  
            translated_chunk = translator.translate(chunk) 
            translated_chunks.append(translated_chunk)
    return ' '.join(translated_chunks)
batch_size = 100  
num_batches = len(data) // batch_size + 1

translated_descriptions = []
for i in range(num_batches):
    print(f"Translating batch {i+1}/{num_batches}")
    batch = data['Description'][i*batch_size:(i+1)*batch_size]
    translated_batch = [translate_text(text) for text in batch]
    translated_descriptions.extend(translated_batch)
    time.sleep(1)  

data['Description_ko'] = translated_descriptions
data['Severity_ko'] = data['Severity']

data['combined_text_ko'] = "Description: " + data['Description_ko'] + " Priority: " + data['Severity_ko'].astype(str)

data.to_csv('./data/sev_translated.csv', index=False)

print(data[['Description_ko', 'Severity_ko', 'combined_text_ko']].head())
