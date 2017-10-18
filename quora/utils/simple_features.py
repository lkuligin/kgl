#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding utf-8

import numpy as np
import re
import pandas as pd
import os, sys
from encoder import parse_line
from utils import load_data, save_data, fit_min_max_scale
import argparse

pattern_cap = re.compile(r'[A-Z]')
pattern_capb = re.compile(r'^[A-Z]')
pattern_dig = re.compile(r'[0-9]')
pattern_digf = re.compile(r'^[0-9]$')
pattern_unic = re.compile(r'\x00-\x7f')
pattern_unicf = re.compile(r'^\x00-\x7f$')

STRING_LAMBDAS = {'len': lambda x: len(x),
  'cap': lambda x: len(re.findall(pattern_cap,x)),
  'digits': lambda x: len(re.findall(pattern_dig,x)),
  'cap_big': lambda x: len(re.findall(pattern_capb,x)),
  'unic': lambda x: len(re.findall(pattern_unic,x))
  }

TOKENS_LAMBDAS = {'len': lambda x: len(x),
  'unique_len': lambda x: len(set(x)),
  'cap': lambda x: len([1 for el in x if re_match(el, pattern_cap)]),
  'cap_big': lambda x: len([1 for el in x if re_match(el, pattern_capb)]),
  'digits': lambda x: len([1 for el in x if re_match(el, pattern_dig)]),
  'digits_full': lambda x: len([1 for el in x if re_match(el, pattern_dig)]),
  'avg_len': lambda x: sum([len(el) for el in x])/len(x) if len(x)>0 else 0
  }

PAIR_TOKENIZE_FEATURES = {
  'rel_jackard_mod': lambda tokens1, tokens2: jackard_dist_mod(set(tokens1), set(tokens2))
  }

HOT_ENC_TOKENS = {
  'same_first':  lambda x, y: 1 if x[0] ==y[0] else 0
}

def parse_args():
  #TODO add argument for test/train dataset
  parser = argparse.ArgumentParser(description='Creates vectorizer')
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='submit preparation')

  return parser.parse_args()

def re_match(string, pattern):
  return True if pattern.match(string) == None else False

def compute_ratio(val1, val2, val = None):
  if val and val > 0:
    return abs(val1-val2)/val
  elif val1 == 0 or val2 == 0:
    return 0
  return 1.*min(val1, val2)/max(val1, val2)

def compute_feature(val1, val2, f):
  el1 = f(val1)
  el2 = f(val2)
  return [max(el1, el2), min(el1, el2), compute_ratio(el1, el2), el1+el2, abs(el1-el2)]

def jackard_dist_mod(set1, set2):
  u = set1.union(set2)
  if len(u) == 0:
    return 0
  return 2.*len(set1.intersection(set2))/(len(set1)+len(set2))

def add_features_lambdas(df, field1_name, field2_name, dict, dict_name):
  for feature, func in dict.items():
    df["dist_max_{0}_{1}".format(feature, dict_name)] = df.apply(lambda row: compute_feature(row[field1_name], row[field2_name], func)[0], axis = 1)
    df["dist_min_{0}_{1}".format(feature, dict_name)] = df.apply(lambda row: compute_feature(row[field1_name], row[field2_name], func)[1], axis = 1)
    df["dist_rat_{0}_{1}".format(feature, dict_name)] = df.apply(lambda row: compute_feature(row[field1_name], row[field2_name], func)[2], axis = 1)
    df["dist_sum_{0}_{1}".format(feature, dict_name)] = df.apply(lambda row: compute_feature(row[field1_name], row[field2_name], func)[3], axis = 1)
    df["dist_dif_{0}_{1}".format(feature, dict_name)] = df.apply(lambda row: compute_feature(row[field1_name], row[field2_name], func)[4], axis = 1)


def scale(df, train = True):
  for d, dict_name in [(STRING_LAMBDAS, "smpl"), (TOKENS_LAMBDAS, "tkn")]:
    for feature in d.keys():
      for prefix in ["max", "min", "sum", "dif"]:
        feature_name = "dist_{0}_{1}_{2}".format(prefix, feature, dict_name)
        if train:
          df[feature_name] = fit_min_max_scale(df[feature_name].values.reshape(-1,1))

def add_features(df):
  for feature, func in PAIR_TOKENIZE_FEATURES.items():
    df[feature] = df.apply(lambda row: func(row["question1_tk"], row["question1_tk"]), axis = 1)

def main():
  args = parse_args()
  train = True
  initial = False
  if args.fin:
    train = False
  print("preparing data for train: ", train)

  data = load_data(train, initial)
  add_features_lambdas(data, 'question1', 'question2', STRING_LAMBDAS, "smpl")
  add_features_lambdas(data, 'question1_tk', 'question2_tk', TOKENS_LAMBDAS, "tkn")
  add_features(data)
  print("features added!")
  scale(data)
  print("data scaled!")
  #save_data(data, train)

if __name__ == '__main__':
  main()