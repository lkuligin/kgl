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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from xgb import rescale
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from encoder import parse_line

PARAMS_ = {
  ("cv", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("cv", 1, "sgd", "diff"): {"alpha": 0.01, "l1_ratio": 0.0001},
  ("cv", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("cv", 2, "sgd", "diff"): {"alpha": 0.0001, "l1_ratio": 0.0001},
  ("cv", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("cv", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tf", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.9},
  ("tf", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tf", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf", 1, "sgd", "intersect"): {"alpha": 0.0001, "l1_ratio": 0.0001},
  ("tfidf", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tfidf", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tfidf", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("cv_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.001},
  ("cv_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.001},
  ("cv_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tfidf_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tf_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tf_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tf_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0}
}

PARAMS = {
  ("cv", 1, "nb", "intersect"): {"alpha": 0.5, "fit_prior": True},
  ("cv", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("cv", 1, "nb", "diff"): {"alpha": 0.9, "fit_prior": True},
  ("cv", 1, "sgd", "diff"): {"alpha": 0.01, "l1_ratio": 0.0001},
  ("cv", 2, "nb", "intersect"): {"alpha": 0.5, "fit_prior": True},
  ("cv", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("cv", 2, "nb", "diff"): {"alpha": 0.9, "fit_prior": True},
  ("cv", 2, "sgd", "diff"): {"alpha": 0.0001, "l1_ratio": 0.0001},
  ("cv", 3, "nb", "intersect"): {"alpha": 0.9, "fit_prior": True},
  ("cv", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("cv", 3, "nb", "diff"): {"alpha": 0.9, "fit_prior": True},
  ("cv", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tf", 1, "nb", "intersect"): {"alpha": 0.5, "fit_prior": True},
  ("tf", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.9},
  ("tf", 1, "nb", "diff"): {"alpha": 0.01, "fit_prior": True},
  ("tf", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tf", 2, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tf", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf", 2, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tf", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf", 3, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tf", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf", 1, "nb", "intersect"): {"alpha": 0.9, "fit_prior": True},
  ("tfidf", 1, "sgd", "intersect"): {"alpha": 0.0001, "l1_ratio": 0.0001},
  ("tfidf", 1, "nb", "diff"): {"alpha": 0.01, "fit_prior": True},
  ("tfidf", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tfidf", 2, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf", 2, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.01},
  ("tfidf", 3, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf", 3, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("cv_t", 1, "nb", "intersect"): {"alpha": 0.1, "fit_prior": True},
  ("cv_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.001},
  ("cv_t", 1, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("cv_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 2, "nb", "intersect"): {"alpha": 0.01, "fit_prior": True},
  ("cv_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 2, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("cv_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("cv_t", 3, "nb", "intersect"): {"alpha": 0.01, "fit_prior": True},
  ("cv_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 0.001},
  ("cv_t", 3, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("cv_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tfidf_t", 1, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 1, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf_t", 2, "nb", "intersect"): {"alpha": 0.5, "fit_prior": False},
  ("tfidf_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 2, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tfidf_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tfidf_t", 3, "nb", "intersect"): {"alpha": 0.1, "fit_prior": False},
  ("tfidf_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tfidf_t", 3, "nb", "diff"): {"alpha": 0.0001, "fit_prior": False},
  ("tfidf_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0},
  ("tf_t", 1, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tf_t", 1, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 1, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tf_t", 1, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tf_t", 2, "nb", "intersect"): {"alpha": 0.0001, "fit_prior": True},
  ("tf_t", 2, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 2, "nb", "diff"): {"alpha": 0.0001, "fit_prior": True},
  ("tf_t", 2, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0.0001},
  ("tf_t", 3, "nb", "intersect"): {"alpha": 0.1, "fit_prior": False},
  ("tf_t", 3, "sgd", "intersect"): {"alpha": 1e-05, "l1_ratio": 1},
  ("tf_t", 3, "nb", "diff"): {"alpha": 0.0001, "fit_prior": False},
  ("tf_t", 3, "sgd", "diff"): {"alpha": 1e-05, "l1_ratio": 0}
}

def boost_SGD(X, Y, nfolds=10):
    classifier = lambda: SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=100, 
        shuffle=True, 
        n_jobs=-1,
        class_weight=None)

    steps = [('clf', classifier())]
    model = Pipeline(steps=steps)

    parameters = {
        'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.5, 0.9, 1],
        'clf__l1_ratio': [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 1]
    }
    
    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=nfolds, 
        n_jobs=-1,
        verbose=1)
    
    grid_search = grid_search.fit(X, Y)
    print(grid_search.best_params_)

def boost_NV(X, Y, nfolds=10):
    classifier = lambda: MultinomialNB()

    steps = [('clf', classifier())]
    model = Pipeline(steps=steps)

    parameters = {
        'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-1, 0.5, 0.9],
        'clf__fit_prior': [True, False]
    }
    
    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=nfolds, 
        n_jobs=-1,
        verbose=1)
    
    grid_search = grid_search.fit(X, Y)
    print(grid_search.best_params_)

def parse_args():
  #TODO add argument for test/train dataset
  parser = argparse.ArgumentParser(description='Creates vectorizer')
  parser.add_argument('-c', '--cnt', action='store_const', const=True, help='count vectorizer')


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

def stack(data_train):
  data = rescale(data_train)
  for vect_type in ["cv_t", "tfidf_t", "tf_t"]:
    for i in range(3):
      cv = create_cv(data_train, i+1, vect_type)
      q1 = cv.transform(data.question1.values)
      q2 = cv.transform(data.question2.values)
      print("ngrams=", i+1, " vect_type=", vect_type)
      intersect = q1.minimum(q2)
      print(intersect.shape)
      
      print("intersect")
      print("naive bayes")
      boost_NV(intersect, data.is_duplicate.values) 
      print("SGD")
      boost_SGD(intersect, data.is_duplicate.values)
      
      print("diff")
      diff = q1.maximum(q2) - intersect
      print("SGD")
      boost_SGD(diff, data.is_duplicate.values)
      print("naive bayes")
      boost_NV(diff, data.is_duplicate.values) 

def train_model(vector):
  q1 = cv.transform(data.question1.values)
  q2 = cv.transform(data.question2.values)

def features(data_train, data_test):
  vect_name = None
  vect = None
  ngrams = None
  q1_train = None
  q2_train = None
  q1_test = None
  q2_test = None
  for (vect_next, ngrams_next, model_type, vector_type), params in PARAMS.items():
    if (vect_next != vect_name) | (ngrams_next != ngrams):
      vect_name = vect_next
      ngrams = ngrams_next
      
      vect = create_cv(data_train, ngrams, vect_name)
      q1_train = vect.transform(data_train.question1.values)
      q2_train = vect.transform(data_train.question2.values)
      q1_test = vect.transform(data_test.question1.values)
      q2_test = vect.transform(data_test.question2.values)

    vector_train = q1_train.minimum(q2_train) if vector_type == "intersect" else q1_train.maximum(q2_train) - q1_train.minimum(q2_train)
    vector_test = q1_test.minimum(q2_test) if vector_type == "intersect" else q1_test.maximum(q2_test) - q1_test.minimum(q2_test)
    
    model = None
    if model_type == "nb":
      model = MultinomialNB(alpha = params["alpha"], fit_prior = params["fit_prior"])
      model.fit(vector_train, data_train.is_duplicate.values) 
    elif model_type == "sgd":
      model = SGDClassifier(loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=200, 
        shuffle=True, 
        n_jobs=-1,
        class_weight=None,
        alpha=params["alpha"],
        l1_ratio=params["l1_ratio"]
      )
      model.fit(vector_train, data_train.is_duplicate.values)
      
    predict_train = model.predict_proba(vector_train)[:, 0]
    predict_test = model.predict_proba(vector_test)[:, 0]

    col_name = "_{0}_n{1}_{2}_{3}_".format(vect_name, ngrams, model_type, vector_type)
    data_train[col_name] =  predict_train
    data_test[col_name] = predict_test
    print(col_name)

  save_data(data_train, True)
  save_data(data_test, False)
    



def main():
  data_train = load_data(True, False)
  data_test = load_data(False, False)
  features(data_train, data_test)

  #stack(data_train)
  
if __name__ == '__main__':
  main()

