#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding utf-8

import numpy as np
import re
import pandas as pd
import os, sys
from encoder import parse_line
from utils import load_data, save_data
from sklearn.preprocessing import MinMaxScaler
import argparse

PAIR_FEATURES = {'rel_simple_len': lambda text1, text2: compute_ratio(len(text1), len(text2)),
  'rel_capitalized': lambda text1, text2: compute_ratio(len(re.findall(r'[A-Z]',text1)), len(re.findall(r'[A-Z]',text2))),
  'rel_capitalized_mod': lambda text1, text2: compute_ratio(len(re.findall(r'[A-Z]',text1)), len(re.findall(r'[A-Z]',text2)), max(len(text1), len(text2))),
  'rel_digits': lambda text1, text2: compute_ratio(len(re.findall(r'[0-9]',text1)), len(re.findall(r'[0-9]',text2))),
  'rel_digits_mod': lambda text1, text2: compute_ratio(len(re.findall(r'[0-9]',text1)), len(re.findall(r'[0-9]',text2)), max(len(text1), len(text2)))
  }
SIMPLE_FEATURES = {'simple_len': lambda text: len(text),
  'cap_len': lambda text: 1.*len(re.findall(r'[A-Z]',text))/len(text) if len(text) >0 else 0,
  'digits_len': lambda text: 1.*len(re.findall(r'[0-9]',text))/len(text) if len(text) > 0 else 0
  }
SIMPLE_TOKENIZE_FEATURES = {
  'tokens_len': lambda tokens: len(tokens),
  'tokens_unique_len': lambda tokens: len(set(tokens))
  }
PAIR_TOKENIZE_FEATURES = {
  'token_list_rel': lambda tokens1, tokens2: compute_ratio(len(tokens1), len(tokens2)),
  'rel_jackard': lambda tokens1, tokens2: jackard_dist(set(tokens1), set(tokens2)),
  'rel_jackard_mod': lambda tokens1, tokens2: jackard_dist_mod(set(tokens1), set(tokens2))
  }

def parse_args():
  #TODO add argument for test/train dataset
  parser = argparse.ArgumentParser(description='Creates vectorizer')
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='submit preparation')

def compute_ratio(val1, val2, val = None):
  if val and val > 0:
    return abs(val1-val2)/val
  elif val1 == 0 or val2 == 0:
    return 0
  return 1.*min(val1, val2)/max(val1, val2)


def jackard_dist(set1, set2):
  u = set1.union(set2)
  if len(u) == 0:
    return 0
  return 1.*len(set1.intersection(set2))/len(set1.union(set2))

def jackard_dist_mod(set1, set2):
  u = set1.union(set2)
  if len(u) == 0:
    return 0
  return 2.*len(set1.intersection(set2))/(len(set1)+len(set2))

def add_features(df, field1_name, field2_name, dict1 = PAIR_FEATURES, dict2 = SIMPLE_FEATURES):
  for feature, func in dict1.items():
    df[feature] = df.apply(lambda row: func(row[field1_name], row[field2_name]), axis = 1)
  for feature, func in dict2.items():
    df[''.join((feature,'1'))] = df.apply(lambda row: func(row[field1_name]), axis = 1)
    df[''.join((feature,'2'))] = df.apply(lambda row: func(row[field2_name]), axis = 1)

def scale_column(df, col1, col2):
  scaler = MinMaxScaler()
  cnct = np.concatenate([df[col1].values,df[col2].values])
  scaler = scaler.fit(cnct.reshape(-1,1))
  df[col1] = scaler.transform(df[col1].values.reshape(-1,1))
  df[col2] = scaler.transform(df[col2].values.reshape(-1,1))

def scale(df):
  df['avg_length1'] = (df.simple_len1.astype(float)/df.tokens_len1).replace([np.inf, -np.inf, np.nan], 0)
  df['avg_length2'] = (df.simple_len2.astype(float)/df.tokens_len2).replace([np.inf, -np.inf, np.nan], 0)
  df['avg_length'] = df.apply(lambda row: compute_ratio(row['avg_length1'], row['avg_length2']), axis = 1)
  scale_column(df, 'simple_len1', 'simple_len2')
  scale_column(df, 'tokens_len1', 'tokens_len2')
  scale_column(df, 'tokens_unique_len1', 'tokens_unique_len1')
  scale_column(df, 'avg_length1', 'avg_length2')
  

def add_other_features(df):
  cosine = lambda row: row['simple_dot']/row['simple_norm1']/row['simple_norm2'] if (row['simple_norm1'] > 0 and row['simple_norm2']>0) else 0
  #tfidf_cosine = lambda row: row['tfidf_dot']/row['tfidf_norm1']/row['tfidf_norm2'] if (row['tfidf_norm1']>0 and row['tfidf_norm2']>0) else 0
  df['simle_cosine'] = df.apply(cosine, axis = 1)
  #df['tfidf_cosine'] = df.apply(tfidf_cosine, axis = 1)

def main():
  args = parse_args()
  train = True
  if args.f:
    train = False
  print("preparing data for train: ", train)
  initial = False

  data = load_data(train, initial)
  add_features(data, 'question1', 'question2')
  add_features(data, 'question1_tk', 'question2_tk', PAIR_TOKENIZE_FEATURES, SIMPLE_TOKENIZE_FEATURES)
  add_other_features(data)
  scale(data)
  save_data(data, train)

if __name__ == '__main__':
  main()