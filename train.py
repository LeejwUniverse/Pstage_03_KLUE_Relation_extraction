import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  # load model and tokenizer
  MODEL_NAME = "bert-base-multilingual-cased"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("./dataset/train.tsv")
  dev_dataset = load_data("./dataset/dev.tsv")
  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  bert_config = BertConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  model = BertForSequenceClassification(bert_config) 
  model.parameters
  model.to(device)

  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()