#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import load_data, save_data
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from encoder import parse_line


def parse_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='enrich test data')
  return parser.parse_args()

def create_cv(data, i, name="cv"):
  vect = CountVectorizer(input='content'
    , strip_accents = 'unicode'
    , analyzer = 'char'
    , ngram_range=(i, i)
  )
  if name == "tf":
    vect = TfidfVectorizer(input='content'
      , strip_accents = 'unicode'
      , analyzer = 'char'
      , norm = 'l2'
      , use_idf = False
      , sublinear_tf = False
      , ngram_range=(i, i)
    )
  elif name =="tfidf":
    vect = TfidfVectorizer(input='content'
      , strip_accents = 'unicode'
      , analyzer = 'char'
      , norm = 'l2'
      , use_idf = True
      , smooth_idf = True
      , sublinear_tf = False
      , ngram_range=(i, i)
    )
  elif name == "cv_t":
    vect = CountVectorizer(input='content'
      , strip_accents = 'unicode'
      , analyzer = 'word'
      , tokenizer = parse_line
      , ngram_range=(i, i)
  )
  elif name == "tf_t":
    vect = TfidfVectorizer(input = 'content'
      , strip_accents = 'unicode' #'ascii' #None 
      , analyzer = 'word'
      , tokenizer = parse_line
      , use_idf = False
      , sublinear_tf = False
      , norm = 'l2'
      , ngram_range=(i, i)
  )
  elif name == "tfidf_t":
    vect = TfidfVectorizer(input = 'content'
      , strip_accents = 'unicode' #'ascii' #None 
      , analyzer = 'word'
      , tokenizer = parse_line
      , use_idf = True
      , smooth_idf = True
      , sublinear_tf = False
      , norm = 'l2'
      , ngram_range=(i, i)
  )

  all_questions = pd.concat([data.question1, data.question2]).unique()
  vect.fit(all_questions)
  return vect

def add_features(data, cv, i, vect_name = "cv"):
  q1 = cv.transform(data.question1.values)
  q2 = cv.transform(data.question2.values)

  #count of ngrams
  data["_{0}_cnt_l_{1}_q1".format(vect_name, i)]=q1.sum(axis=1).A.flatten()
  data["_{0}_cnt_l_{1}_q2".format(vect_name, i)]=q2.sum(axis=1).A.flatten()

  #count unique ngrams
  data["_{0}_cnt_lu_{1}_q1".format(vect_name, i)]=(q1>0).sum(axis=1).A.flatten()
  data["_{0}_cnt_lu_{1}_q2".format(vect_name, i)]=(q2>0).sum(axis=1).A.flatten()

  #intersect ngrams
  intersect=q1.minimum(q2).sum(axis=1).A.flatten()
  union=q1.maximum(q2).sum(axis=1).A.flatten()
  data["_{0}_jack_l_{1}".format(vect_name, i)]=intersect/union

  #intersect unqiue ngrams
  intersect=q1.minimum(q2).sum(axis=1).A.flatten()
  union=q1.maximum(q2).sum(axis=1).A.flatten()
  data["_{0}_jack_l_{1}".format(vect_name, i)]=intersect/union

def train_svd(data, cv, i, vect_name = "cv"):
  q1 = cv.transform(data.common.values)
  q2 = cv.transform(data.diff.values)


def main():
  args = parse_args()
  train = True
  data_train = load_data(True, False)
  data = data_train
  if args.fin:
    train = False
    data = load_data(train, False)

  #data_train['common'] = data_train.apply(lambda row: list((Counter(row.question1_tk) & Counter(row.question2_tk)).elements()))
  #data_train['diff'] = data_train.apply(lambda row: list(((Counter(row.question1_tk) | Counter(row.question2_tk) - (Counter(row.question1_tk) & Counter(row.question2_tk)).elements()))#

  for vect_type in ["cv", "cv_t", "tf", "tf_t", "tfidf", "tfidf_t"]:
    for i in range(3):
      ngrams = i+1
      cv = create_cv(data_train, ngrams, vect_type)
      add_features(data, cv, ngrams, vect_type)
  
  save_data(data, train)
  
if __name__ == '__main__':
  main()

