#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding=utf-8

import pickle
import argparse
import pandas as pd
import numpy as np
from utils import load_data, save_data, fit_min_max_scale
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from encoder import parse_line, STOPWORDS, stemmer, PATH_COUNT_VECTORIZER, PATH_TFIDF_VECTORIZER
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
from sklearn.preprocessing import MinMaxScaler

DISTANCES = ['braycurtis', 'canberra', 'cosine', 'jaccard', 'l1', 'l2']
#correlation, yule, chebychev, 

def parse_args():
  #TODO add argument for test/train dataset
  parser = argparse.ArgumentParser(description='Creates vectorizer')
  parser.add_argument('-c', '--cnt', action='store_const', const=True, help='count vectorizer')
  parser.add_argument('-t', '--tf', action='store_const', const=True, help='tfidf vectorizer')
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='submit preparation')
  
  return parser.parse_args()

def compute_distances(vectorizer, data, vectorizer_name = "", train = True, col1_name = "question1", col2_name = "question2"):
  vector1 = vectorizer.transform(data[col2_name].values)
  vector2 = vectorizer.transform(data[col1_name].values)
  print(vectorizer_name, " data prepared")
  for distance_type in DISTANCES:
    metric_name = "distance_{0}_{1}".format(vectorizer_name, distance_type)
    metric_values = np.array([])
    if distance_type in ['cosine', 'l1', 'l2']:
      metric_values = paired_distances(vector1, vector2, metric = distance_type)
    else:
      metric = lambda x, y: pairwise_distances(x.reshape(1,-1), y.reshape(1,-1), metric = distance_type)
      for el1, el2 in zip(vector1, vector2):
        metric_values = np.append(metric_values, metric(el1.toarray(), el2.toarray()))
    if distance_type in ['canberra', 'l1']:
      if train:
        print(metric_name)
        metric_values = fit_min_max_scale(metric_values.reshape(-1, 1)).flatten()
      else:
        metric_values = min_max_scale(metric_values.reshape(-1, 1), metric_name)
    data[metric_name] = metric_values
    print("metric {0}, minimum {1:.4f}, maximum {2:.4f}".format(distance_type, np.min(metric_values), np.max(metric_values)))

def compute_distances_from_vectors(vector1, vector2, vectorizer_name = "", rescale = False):
  scaler = MinMaxScaler()
  res = pd.DataFrame()
  print("starting distance calculation")
  for distance_type in DISTANCES:
    metric = lambda x, y: pairwise_distances(x.reshape(1,-1), y.reshape(1,-1), metric = distance_type)
    metric_name = "distance_{0}_{1}".format(vectorizer_name, distance_type)
    metric_value = paired_distances(vector1, vector2, metric = metric)
    if distance_type in ['canberra', 'l1']:
        metric_value = scaler.fit_transform(metric_value.reshape(-1, 1)).flatten()
    res[metric_name] = metric_value
  return res

def main():
  args = parse_args()
  train = True
  if args.fin:
    train = False
  print("preparing data for train: ", train)

  data = load_data(train, False)
  if args.cnt:
    cv  = pickle.load(open(PATH_COUNT_VECTORIZER, "rb"))
    print("cv vectorizer loaded")
    compute_distances(cv, data, "cv")
  if args.tf:
    tf  = pickle.load(open(PATH_TFIDF_VECTORIZER, "rb"))
    print("tf vectorizer loaded")
    compute_distances(tf, data, "tf")
  save_data(data, train)    



if __name__ == "__main__":
  main()
