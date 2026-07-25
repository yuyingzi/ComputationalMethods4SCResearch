from pathlib import Path

import torch
from datasets import load_dataset as load_hf_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, set_seed
from torch.utils.data import Dataset, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir = Path(__file__).resolve().parent

class CustomDataset(Dataset):
    """A custom dataset class for your irony detection task."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_irony_dataset(file_path, tokenizer, max_length):
    """Function to load and tokenize the dataset."""
    df = pd.read_csv(file_path)
    texts = df['tweet'].tolist()
    labels = df['sarcastic'].tolist()

    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return CustomDataset(encodings, labels)

def compute_metrics(pred):
    """Function to compute metrics of the model's performance."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load and tokenize the dataset bundled with this repository.
file_path = project_dir / 'combined_new_irony.csv'
max_length = 256
iron_dataset = load_irony_dataset(file_path, tokenizer, max_length)
iron_train_size = int(0.8 * len(iron_dataset))
iron_train_dataset, iron_eval_dataset = random_split(
    iron_dataset,
    [iron_train_size, len(iron_dataset) - iron_train_size],
    generator=torch.Generator().manual_seed(42),
)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir=str(project_dir / 'results' / 'irony'),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(project_dir / 'logs'),
    logging_steps=10,
    eval_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=iron_train_dataset,
    eval_dataset=iron_eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

model_save_path = project_dir / 'iron_model'

# 保存模型和tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
model.eval()

# 对情感分类数据集的每条文本进行反讽预测
def predict_irony_labels(texts, tokenizer, model, device, batch_size=32):
    irony_labels = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            list(texts[start:start + batch_size]),
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Assuming using a binary classification model where the second token (index 1) represents "irony"
        irony_labels.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())

    return irony_labels

# Keep model selection on a validation split and reserve the canonical test split.
imdb = load_hf_dataset("stanfordnlp/imdb")
sentiment_split = imdb["train"].train_test_split(test_size=0.1, seed=42)
train_texts, train_labels = sentiment_split["train"]["text"], sentiment_split["train"]["label"]
eval_texts, eval_labels = sentiment_split["test"]["text"], sentiment_split["test"]["label"]
test_texts, test_labels = imdb["test"]["text"], imdb["test"]["label"]

# 假设你的模型和tokenizer已经定义好了
train_irony_labels = predict_irony_labels(train_texts, tokenizer, model, device)
eval_irony_labels = predict_irony_labels(eval_texts, tokenizer, model, device)
test_irony_labels = predict_irony_labels(test_texts, tokenizer, model, device)

class CustomDataset1(Dataset):
    def __init__(self, texts, labels, irony_labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.irony_labels = irony_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        prefix = "[IRONY]" if self.irony_labels[idx] else "[NON_IRONY]"
        item = self.tokenizer(f"{prefix} {self.texts[idx]}", truncation=True, padding="max_length",
                              max_length=self.max_length, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in item.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.texts)

tokenizer.add_special_tokens({'additional_special_tokens': ['[IRONY]', '[NON_IRONY]']})
train_dataset = CustomDataset1(train_texts, train_labels, train_irony_labels, tokenizer, 256)
eval_dataset = CustomDataset1(eval_texts, eval_labels, eval_irony_labels, tokenizer, 256)
test_dataset = CustomDataset1(test_texts, test_labels, test_irony_labels, tokenizer, 256)

model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
model1.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir=str(project_dir / 'results' / 'sentiment'),
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(project_dir / 'logs'),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    # Add the following line to report metrics every evaluation step
    report_to="none"
)

# Initialize the Trainer
trainer1 = Trainer(
    model=model1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer1.train()

trainer1.evaluate(test_dataset)
