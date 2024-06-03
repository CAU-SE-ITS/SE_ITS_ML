import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.optim as optim
from tqdm import tqdm
from kobert_transformers import get_tokenizer, get_kobert_model

# CSV 파일 로드
data = pd.read_csv('./data/sev_translated.csv')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'(Created by [\w\s]+ on [\w\s]+ \d+ \d+)', '', text)  # 'Created by ...' 패턴 제거
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)  # 날짜 패턴 제거 (예: 1998-04-07)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # 이메일 패턴 제거
    return text

data['description'] = data['Description'].apply(preprocess_text)
data = data.rename(columns={'Severity': 'priority'})

data['combined_text'] = "Description: " + data['Description'] + " Priority: " + data['priority'].astype(str)

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

tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
model = AutoModel.from_pretrained('monologg/kobert')

# 데이터셋 생성
train_texts = data['combined_text'].tolist()

train_dataset = CustomDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    max_len=512
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

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


    print(f"Epoch {epoch+1} completed")

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
