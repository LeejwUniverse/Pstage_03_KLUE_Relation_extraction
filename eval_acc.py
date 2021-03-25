import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

def main():
  pred = pd.read_csv("./prediction/pred_answer.csv")
  answer = pd.read_csv("./answer/answer.csv")
  acc = accuracy_score(list(pred["pred"]), list(answer["answer"]))
  print("accuracy: ",acc)

if __name__ == '__main__':
  main()