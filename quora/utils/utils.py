# coding utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

PATH_ENRICHED_TEST_DATA = '../data/test.pickle'
PATH_ENRICHED_TRAIN_DATA = '../data/train.pickle'

SCORE_SUBMIT_0 = 6.01888
SCORE_SUBMIT_05 =  0.69315
SCORE_SUBMIT_1 = 28.52056

SCALERS = {
  
}

def fit_min_max_scale(arr, arr1 = None):
  scaler = MinMaxScaler()
  res =  scaler.fit_transform(arr)
  if arr1 == None:
    print(scaler.get_params())
    return res
  else:
    return res, scaler.tranform(arr1)

def adj_coeff():
  a = np.log(0.5)*(SCORE_SUBMIT_0 + SCORE_SUBMIT_1)/SCORE_SUBMIT_05
  e = (1 - np.sqrt(1 - 4*np.exp(a)))/2
  y1 = (SCORE_SUBMIT_1 - np.log(e)/np.log(1-e)*SCORE_SUBMIT_0) / (np.log(e)**2/np.log(1-e) - np.log(1-e))
  y0 = 1-y1
  return (y0, y1)

def submit_const_score(score):
  data = load_data(False, True)
  res = pd.DataFrame()
  res['test_id'] = data['test_id']
  res['is_duplicate'] = score
  res.to_csv('../data/v0.csv', index=False)

class AdjustScore:
  def __init__(self, data=None):
    c0, c1 = adj_coeff()
    self.c0 = 1.309
    self.c1 = 0.472
    if data:
      y_train = data.shape[0]
      y0_train = data[data.is_duplicate == 0].shape[0] / y_train
      y1_train = data[data.is_duplicate == 1].shape[0] / y_train
      self.c0 = 1.0*c0/y0_train
      self.c1 = 1.0*c1/y1_train

  def adjust(self, score):
    return self.c1 * score / (self.c1 * score + self.c0 * (1-score))

class CrossVal:
  """iterable that yields (train, test) split
  """
  def __init__(self, data, n=10):
    self.n = n
    self.y = data.is_duplicate.values
    self.x = data.index.values

    c0, c1 = adj_coeff()
    y = data.shape[0]
    y1 = data[data.is_duplicate == 1].shape[0]
    y0 = data[data.is_duplicate == 0].shape[0]
    split = 1. - 1./n

    self.size_train0 = int(y0 * split)  #zeros in a train split for each iteration
    self.size_train1 = int(self.size_train0 * y1/y0)  #ones in a train split for each iteration, same proportion as actual train
    self.size_test1 =  int((y-self.size_train0)* c1/c0)  #ones in a train split for each iteration, same proportion as actual test

  def __iter__(self):
    train0 = np.random.choice(self.x, self.size_train0, replace = False)
    train1 = np.random.choice(self.y, self.size_train1, replace = False) # train = same proportion as in train
    test0 = np.setdiff1d(self.x, train0)
    test1 = np.random.choice(np.setdiff1d(self.y, train1), self.size_test1, replace = False) # test = same proportion as in test

    train = np.hstack((train0, train1))
    test = np.hstack((test0, test1))

    yield np.shuffle(train), np.shuffle(test)

class SvdExplain:
    def __init__(self, svd, tfv):
        self.components = svd.components_
        self.features = tfv.get_feature_names()
        self.n_components = svd.get_params()['n_components']
    
    def explain(self, ind, n_tokens = 20):
        amount = min(n_tokens, self.n_components)
        components = self.components[ind]
        components = np.sort(components)[-amount:][::-1]
        for val in components:
            feature_ind = np.where(self.components[ind] == val)[0][0]
            print(val, feature_ind, self.features[feature_ind])

def load_data(train = True, initial = False):
  #TODO move to separate Class
  data = None
  if train:
    data = pd.read_csv('../data/train.csv') if initial else pickle.load(open(PATH_ENRICHED_TRAIN_DATA, "rb"))
  else:
    data = pd.read_csv('../data/test.csv') if initial else pickle.load(open(PATH_ENRICHED_TEST_DATA, "rb"))  
  data.question1 = data.question1.fillna("")
  data.question2 = data.question2.fillna("")
  return data

def save_data(data, train = True):
  if train:
    pickle.dump(data, open(PATH_ENRICHED_TRAIN_DATA, 'wb'))
  else:
    print(PATH_ENRICHED_TEST_DATA)
    pickle.dump(data, open(PATH_ENRICHED_TEST_DATA, 'wb'))

if __name__ == "__main__":
  print("test")
  #submit_const_score(1.0)
