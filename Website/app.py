import torch
from flask import Flask, request, render_template
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import html  # Import the html library

app = Flask(__name__)

# Load the trained model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.load_state_dict(torch.load('models/roberta_model.pt', map_location=device))
model.to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load feedback data
feedback_file = 'feedback.csv'
if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
    feedback_data = pd.read_csv(feedback_file, usecols=['news_title', 'label'], encoding='utf-8')
else:
    feedback_data = pd.DataFrame(columns=['news_title', 'label'])

def classify_news(text):
    feedback_row = feedback_data[feedback_data['news_title'] == text]
    if not feedback_row.empty:
        pred = feedback_row['label'].values[0]
        label_map = {0: 'Fake News', 1: 'Real News', 2: 'AI-Generated News'}
        return label_map[pred]
    else:
        tokens = tokenizer.tokenize(text)[:512-2]
        tokens = ['<s>'] + tokens + ['</s>']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tensor = torch.tensor(ids).unsqueeze(0).to(device)
        mask = torch.ones_like(tensor).to(device)
        with torch.no_grad():
            outputs = model(input_ids=tensor, attention_mask=mask)
            logits = outputs.logits
            _, pred = torch.max(logits.data, 1)
            pred = pred.item()
        label_map = {0: 'Fake News', 1: 'Real News', 2: 'AI-Generated News'}
        return label_map[pred]

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(feedback_data, model, tokenizer, device, epochs=1):
    texts = feedback_data['news_title'].tolist()
    labels = feedback_data['label'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
    val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    torch.save(model.state_dict(), 'models/roberta_model.pt')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_title = request.form['news_title']
        prediction = classify_news(news_title)
        return render_template('findex.html', prediction=prediction, news_title=news_title)
    return render_template('findex.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    global feedback_data
    news_title = request.form['news_title']
    news_title = html.unescape(news_title)  # Unescape HTML entities
    correct = request.form.get('correct', 'true') == 'true'
    if not correct:
        label = request.form['label']
        label_map = {v: k for k, v in {0: 'Fake News', 1: 'Real News', 2: 'AI-Generated News'}.items()}
        label_code = label_map[label]
        # Update the feedback_data DataFrame
        feedback_data = pd.concat([feedback_data, pd.DataFrame({'news_title': [news_title], 'label': [label_code]})], ignore_index=True)
        feedback_data.drop_duplicates(subset=['news_title'], keep='last', inplace=True)
        feedback_data.to_csv(feedback_file, index=False, encoding='utf-8')
        train_model(feedback_data, model, tokenizer, device)
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
