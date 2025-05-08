import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset

# Load datasets
train_df = pd.read_csv("data/MELD/train_sent_emo.csv")
dev_df = pd.read_csv("data/MELD/dev_sent_emo.csv")
test_df = pd.read_csv("data/MELD/test_sent_emo.csv")

# Combine train and dev for full training set (optional)
full_train_df = pd.concat([train_df, dev_df])

# Clean the data
def preprocess_data(df):
    # Handle missing values
    df = df.dropna(subset=['Utterance', 'Emotion'])
    
    # Filter only needed columns
    df = df[['Utterance', 'Emotion']]
    
    # Clean text
    df['Utterance'] = df['Utterance'].str.strip()
    df['Utterance'] = df['Utterance'].str.replace(r'\r\n', ' ', regex=True)
    
    return df

train_data = preprocess_data(full_train_df)
test_data = preprocess_data(test_df)

# Encode labels
le = LabelEncoder()
train_data['label'] = le.fit_transform(train_data['Emotion'])
test_data['label'] = le.transform(test_data['Emotion'])

# Save label mapping
label_map = {i: label for i, label in enumerate(le.classes_)}
np.save("data/label_map.npy", label_map)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["Utterance"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Save processed datasets
tokenized_train.save_to_disk("data/processed/train")
tokenized_test.save_to_disk("data/processed/test")