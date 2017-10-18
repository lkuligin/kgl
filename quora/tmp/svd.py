#!/mnt/conda/bin/python
# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append("../utils")
from utils import load_data, save_data
from dist_features import compute_distances_from_vectors
from encoder import parse_line, PATH_COUNT_VECTORIZER, PATH_TFIDF_VECTORIZER, PATH_COUNT_MATRIX, PATH_TFIDF_MATRIX
import xgboost
from xgb import XGB_PARAMS, K_FOLDS, N_ESTIMATORS, FEATURES, train_model, rescale

SVD_COMPONENTS = 300

def train():
  tfifd_matrix = pickle.load(open("tdidf_matrix.pickle", "r"))
  svd = TruncatedSVD(n_components = SVD_COMPONENTS)
  svd_matrix = svd.fit_transform(tfifd_matrix)
  pickle.dump(svd_matrix, open("svd_matrix.pickle", "wb" ))
  pickle.dump(svd, open("svd.pickle", "wb" ))

class Experiment1:
  def __init__(self, data, v_matrix, vect):
    self.matrix = v_matrix
    self.data = data
    self.v1 = vect.transform(data.question1.tolist())
    self.v2 = vect.transform(data.question2.tolist())
    self.bestlogloss = np.inf
    self.bestn = 0

  def run(self, n, explain = False, include = False):
    svd = TruncatedSVD(n_components = n)
    _ = svd.fit_transform(self.matrix)
    print("svd done!")
    v1 = svd.transform(self.v1)
    v2 = svd.transform(self.v2)
    features_add = compute_distances_from_vectors(v1, v2, "svd")
    data_add = pd.concat([self.data, features_add], axis = 1)
    features = FEATURES + list(features_add.columns)
    if include:
      cols1 = ["svd_feature_1_{0}".format(i) for i in range(n)]
      cols2 = ["svd_feature_2_{0}".format(i) for i in range(n)]
      pd1 = pd.DataFrame(v1, columns = cols1)
      data_add = pd.concat([data_add, pd1], axis = 1)
      features = features + cols1
      pd2 = pd.DataFrame(v2, columns = cols2)
      data_add = pd.concat([data_add, pd2], axis = 1)
      features = features + cols2
    data_add = rescale(data_add)
    xgb_data = xgboost.DMatrix(data_add[features].values, label=data_add['is_duplicate'].values)
    print("crossval started!")
    logloss = xgboost.cv(XGB_PARAMS, xgb_data, N_ESTIMATORS, nfold=K_FOLDS, metrics={'logloss'}, seed = 0)['test-logloss-mean'].values[-1:][0]
    print("for n={0} logloss={1:.4f}".format(n, logloss))
    if logloss < self.bestlogloss:
      self.bestlogloss = logloss
      self.bestn = n
    train_model(data_add, False, False, features)

  def save(self):
    pickle.dump(self.data, open("data_mod.pickle", "wb" ))


def check1():
  data = load_data(True, False)
  c = False
  matrix = None
  vect = None
  if c:
    matrix = pickle.load(open(PATH_COUNT_MATRIX, "rb"))
    vect = pickle.load(open(PATH_COUNT_VECTORIZER, "rb"))
  else:
    matrix = pickle.load(open(PATH_TFIDF_MATRIX, "rb"))
    vect = pickle.load(open(PATH_TFIDF_VECTORIZER, "rb"))
  print ("data loaded!")
  experiment = Experiment1(data, matrix, vect)
  experiment.run(SVD_COMPONENTS, True, False)
  #for i in range(2,3):
    #experiment.run(i*10)

def check2():
  data = load_data(True, False)
  print("sample loaded!")

def main():
  check1()

if __name__ == '__main__':
  main()