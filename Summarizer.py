

import os
import random
import numpy as np
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load tokenizer and config
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=None)
config = T5Config.from_pretrained(model_name, cache_dir=None)

# Load and process data
file_path = "/content/drive/MyDrive/Colab Notebooks/dataset"

def load_data(file_path):
    stories = []
    with open(file_path, "r", encoding="utf-8") as f:
           for line in f:
               stories.append(line.strip())

    return stories

train_articles = load_data(os.path.join(file_path, "train_stories.txt"))
train_summaries = load_data(os.path.join(file_path, "train_titles.txt"))

val_articles = load_data(os.path.join(file_path, "val_stories.txt"))
val_summaries = load_data(os.path.join(file_path, "val_titles.txt"))

# Preprocessing function
def preprocess_function(article, summary):
    tokenized_article = tokenizer(article, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    tokenized_summary = tokenizer(summary, truncation=True, max_length=150, padding="max_length", return_tensors="pt")
    return {
        'input_ids': tokenized_article['input_ids'].squeeze(),
        'attention_mask': tokenized_article['attention_mask'].squeeze(),
        'labels': tokenized_summary['input_ids'].squeeze(),
        'summary_attention_mask': tokenized_summary['attention_mask'].squeeze()
    }

# Create datasets
def create_dataset(articles, summaries):
    data = [{'article': article, 'summary': summary} for article, summary in zip(articles, summaries)]
    data_processed = [preprocess_function(article, summary) for article, summary in data]
    data_dict = {
        'input_ids': [item['input_ids'] for item in data_processed],
        'attention_mask': [item['attention_mask'] for item in data_processed],
        'labels': [item['labels'] for item in data_processed],
        'summary_attention_mask': [item['summary_attention_mask'] for item in data_processed]
    }
    return Dataset.from_dict(data_dict)

train_dataset = create_dataset(train_articles, train_summaries)
val_dataset = create_dataset(val_articles, val_summaries)

# Define model
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config).to('cuda')

from transformers import AdamW, get_linear_schedule_with_warmup

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,   # Change batch size to 8 for better GPU utilization
    per_device_eval_batch_size=8,    # Change batch size to 8 for better GPU utilization
    num_train_epochs=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_steps=500,
    logging_dir="./logs",
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,  # Set this to True for mixed precision training
    report_to="none"
)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * training_args.num_train_epochs)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler)
)

# Train the model
trainer.train()

