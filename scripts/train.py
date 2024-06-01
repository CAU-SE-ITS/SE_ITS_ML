import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch
import torch.optim as optim
from tqdm import tqdm

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
data['description'] = data['Description'].apply(preprocess_text)
data = data.rename(columns={'Severity': 'priority'})  # severity를 priority로 변경

# 새로운 텍스트 생성
data['combined_text'] = "Description: " + data['Description'] + " Priority: " + data['priority'].astype(str)

# 커스텀 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 데이터셋 생성
train_texts = data['combined_text'].tolist()

train_dataset = CustomDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    max_len=512
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 옵티마이저 설정
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 훈련 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(5):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩 추출
        
        # 그냥 임베딩 추출을 목적으로 하기 때문에 손실 계산과 역전파 과정은 생략합니다.
        
        # 필요시 임베딩을 사용해 추가 작업 수행
        
    print(f"Epoch {epoch+1} completed")

# 모델 저장
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
