#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sparse
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
from sklearn.linear_model import SGDClassifier

from utils import load_data, save_data
from xgb import rescale
from naive import boost_SGD, create_cv
from dist_features import compute_distances_from_vectors

PARAMS = {
  ("tf", 1, "h"): {'alpha': 1e-05, 'l1_ratio': 0.3},
  ("tf", 1, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf", 2, "h"): {'alpha': 1e-05, 'l1_ratio': 0.5},
  ("tf", 2, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf", 3, "h"): {'alpha': 1e-05, 'l1_ratio': 0.3},
  ("tf", 3, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf", 1, "h"): {'alpha': 0.0001, 'l1_ratio': 0.01},
  ("tfidf", 1, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf", 2, "h"): {'alpha': 0.0001, 'l1_ratio': 0.01},
  ("tfidf", 2, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf", 3, "h"): {'alpha': 1e-05, 'l1_ratio': 0.75},
  ("tfidf", 3, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf_t", 1, "h"): {'alpha': 1e-05, 'l1_ratio': 0.75},
  ("tfidf_t", 1, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf_t", 2, "h"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf_t", 2, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf_t", 3, "h"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tfidf_t", 3, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf_t", 1, "h"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf_t", 1, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf_t", 2, "h"): {'alpha': 1e-05, 'l1_ratio': 0.9},
  ("tf_t", 2, "v"): {'alpha': 1e-05, 'l1_ratio': 1},
  ("tf_t", 3, "h"): {'alpha': 1e-05, 'l1_ratio': 0},
  ("tf_t", 3, "v"): {'alpha': 1e-05, 'l1_ratio': 1}
}

def get_svd_matrix(matrix, matrix1 = None, n=1000):
  svd = TruncatedSVD(n_components = n)
  if matrix1 == None:
    return svd.fit_transform(matrix)
  else:
    matrix0 = svd.fit_transform(matrix)
    return matrix0, svd.transform(matrix1)

def stack_sgd(data_train):
  for vect_name in ["tf", "tdiff", "tfidf_t", "tf_t"]:
    for ngrams in range(3):
      vect = create_cv(data_train, ngrams+1, vect_name)
      #vect = create_cv(data_train, 1, "cv")
      q1 = vect.transform(data_train.question1.values)
      q2 = vect.transform(data_train.question2.values)
      print("ngrams=", ngrams+1, " vect_type=", vect_name)

      svd_matrix1 = get_svd_matrix(sparse.csc_matrix(sparse.hstack((q1, q2))), None, 300)
      boost_SGD(svd_matrix1, data_train.is_duplicate.values, 5)

      svd_matrix2 = get_svd_matrix(sparse.csc_matrix(sparse.vstack((q1, q2))), None,300)
      n=int(svd_matrix2.shape[0]/2)
      svd_q1 = svd_matrix2[:n, :]
      svd_q2 = svd_matrix2[n:, :]
      svd_matrix2_ = svd_q1*svd_q2
      boost_SGD(svd_matrix2_, data_train.is_duplicate.values, 5)

def features(data_train, data_test):
  vect_name = None
  ngrams = None
  vect = None

  for (vect_name_next, ngrams_next, matrix_type), params in PARAMS.items():
    if (vect_name_next != vect_name) | (ngrams_next != ngrams):
      vect_name = vect_name_next
      ngrams = ngrams_next

      vect = create_cv(data_train, ngrams, vect_name)
      q1_train = vect.transform(data_train.question1.values)
      q2_train = vect.transform(data_train.question2.values)
      q1_test = vect.transform(data_test.question1.values)
      q2_test = vect.transform(data_test.question2.values)

    svd_matrix_train = None
    svd_matrix_test = None
    if matrix_type == "h":
      svd_matrix_train, svd_matrix_test = get_svd_matrix(sparse.csc_matrix(sparse.hstack((q1_train, q2_train))), sparse.csc_matrix(sparse.hstack((q1_test, q2_test))), 300)

      model = SGDClassifier(loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=400, 
        shuffle=True, 
        n_jobs=-1,
        class_weight=None,
        alpha=params["alpha"],
        l1_ratio=params["l1_ratio"]
        )
      model.fit(svd_matrix_train, data_train.is_duplicate.values)
    elif matrix_type == "v":
      svd_matrix_train, svd_matrix_test = get_svd_matrix(sparse.csc_matrix(sparse.vstack((q1_train, q2_train))), sparse.csc_matrix(sparse.vstack((q1_test, q1_test))), 300)
      
      n=int(svd_matrix_train.shape[0]/2)
      svd_q1_train = svd_matrix_train[:n, :]
      svd_q2_train = svd_matrix_train[n:, :]
      svd_matrix_train = svd_q1_train*svd_q1_train

      n=int(svd_matrix_test.shape[0]/2)
      svd_q1_test = svd_matrix_test[:n, :]
      svd_q2_test = svd_matrix_test[n:, :]
      svd_matrix_test = svd_q1_test*svd_q2_test

      model = SGDClassifier(loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        n_iter=400, 
        shuffle=True, 
        n_jobs=-1,
        class_weight=None,
        alpha=params["alpha"],
        l1_ratio=params["l1_ratio"]
        )
      model.fit(svd_matrix_train, data_train.is_duplicate.values)

      col_name = "_svd_dist_cos_{0}_n{1}_".format(vect_name, ngrams)
      data_train[col_name] = paired_distances(svd_q1_train, svd_q2_train, metric = "cosine")
      data_test[col_name] = paired_distances(svd_q1_test, svd_q2_test, metric = "cosine")

      col_name = "_svd_dist_l1_{0}_n{1}_".format(vect_name, ngrams)
      data_train[col_name] = paired_distances(svd_q1_train, svd_q2_train, metric = "l1")
      data_test[col_name] = paired_distances(svd_q1_test, svd_q2_test, metric = "l1")

      col_name = "_svd_dist_l2_{0}_n{1}_".format(vect_name, ngrams)
      data_train[col_name] = paired_distances(svd_q1_train, svd_q2_train, metric = "l2")
      data_test[col_name] = paired_distances(svd_q1_test, svd_q2_test, metric = "l2")

    predict_train = model.predict_proba(svd_matrix_train)[:, 0]
    predict_test = model.predict_proba(svd_matrix_test)[:, 0]

    col_name = "_svd_{0}_n{1}_{2}_".format(vect_name, ngrams, matrix_type)
    data_train[col_name] =  predict_train
    data_test[col_name] = predict_test
    print(col_name)

  save_data(data_train, True)
  save_data(data_test, False)

def main():
  print("start")
  data_train = load_data(True, False)
  data_test = load_data(False, False)
  #stack_sgd(data_train)
  features(data_train, data_test)

if __name__ == "__main__":
  main()
