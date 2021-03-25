from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np

def pred_answer(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    output_pred.append(result)
    if i==3:
      break
  return list(np.array(output_pred).reshape(-1))

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main():
  """
    test 평가용일 경우 test_datset_dir을 test.tsv로 하면 predicted answer를 얻을 수 있습니다.
    따로 주어진 dev 파일을 활용할 경우, test_datset_dir을 dev.tsv로 바꿔 predicted answer를 얻을 수 있습니다.

    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = "bert-base-multilingual-cased"  
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  MODEL_NAME = "./results/checkpoint-500" # model dir.
  bert_config = BertConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  model = BertForSequenceClassification(bert_config) 
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "./dataset/dev.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  pred_answer = pred_answer(model, test_dataset, device)
  
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv('./prediction/pred_answer.csv', index=False)

if __name__ == '__main__':
  main()