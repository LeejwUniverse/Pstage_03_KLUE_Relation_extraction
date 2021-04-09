import pickle as pickle
import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score

def main(args):
  pred = pd.read_csv(args.pred_answer_dir) # model이 예측한 정답 label
  answer = pd.read_csv(args.public_answer_dir) # test dataset의 정답 label.
  acc = accuracy_score(list(pred["pred"]), list(answer["answer"]))
  print("accuracy: ",acc)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # data dir
  parser.add_argument('--pred_answer_dir', type=str, default="./prediction/pred_answer.csv")
  parser.add_argument('--public_answer_dir', type=str, default="./dataset/test/public/public_answer.csv")
  args = parser.parse_args()
  print(args)
  main(args)
