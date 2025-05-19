import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset

# Load datasets
train_df = pd.read_csv("data/MELD/train_sent_emo.csv")
dev_df = pd.read_csv("data/MELD/dev_sent_emo.csv")
test_df = pd.read_csv("data/MELD/test_sent_emo.csv")

# Combine train and dev
full_train_df = pd.concat([train_df, dev_df])

# Clean the data
def preprocess_data(df):
    df = df.dropna(subset=['Utterance', 'Emotion'])
    df = df[['Utterance', 'Emotion']]
    df['Utterance'] = df['Utterance'].str.strip().str.replace(r'\r\n', ' ', regex=True)
    df['label'] = df['Emotion']
    return df

train_data = preprocess_data(full_train_df)
test_data = preprocess_data(test_df)

# Save label list
label_list = sorted(train_data['label'].unique())
np.save("data/label_list.npy", label_list)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["Utterance"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Save processed data
tokenized_train.save_to_disk("data/processed/train")
tokenized_test.save_to_disk("data/processed/test")
