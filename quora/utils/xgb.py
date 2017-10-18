#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding=utf-8

from utils import load_data, save_data, CrossVal, AdjustScore
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
import argparse
from lasso import FEATURES_NOT_SELECTED, FEATURES_CORR
from sklearn.model_selection import GridSearchCV

SIMPLE_FEATURES = ['rel_digits_mod', 'rel_capitalized_mod', 'rel_capitalized', 'rel_digits', 'rel_simple_len', 'simple_len1', 'simple_len2', \
  'digits_len1', 'digits_len2', 'cap_len1', 'cap_len2', 'token_list_rel', 'tokens_unique_len1', 'tokens_unique_len2', 'tokens_len1', 'tokens_len2', \
  'avg_length1', 'avg_length2', 'avg_length']
DIST_FEATURES1 = ['rel_jackard_mod'] #simple_cosine = distance_cv_cosine, rel_jackard = distance_cv_jaccard
DIST_FEATURES2 = ['distance_cv_braycurtis', 'distance_cv_canberra',  'distance_cv_cosine', 'distance_cv_jaccard', 'distance_cv_l1', 'distance_cv_l2',\
  'distance_tf_canberra', 'distance_tf_l1', 'distance_tf_braycurtis', 'distance_tf_cosine', 'distance_tf_l2'] #distance_tf_jaccard = distance_cv_jaccard

SIMPLE_FEATURES1 = ['dist_max_digits_tkn', 'dist_min_digits_tkn', 'dist_rat_digits_tkn', 'dist_sum_digits_tkn', 'dist_dif_digits_tkn', 'dist_max_len_tkn', \
  'dist_min_len_tkn', 'dist_rat_len_tkn', 'dist_sum_len_tkn', 'dist_dif_len_tkn', 'dist_max_cap_big_tkn', 'dist_min_cap_big_tkn', 'dist_rat_cap_big_tkn', \
  'dist_sum_cap_big_tkn', 'dist_dif_cap_big_tkn', 'dist_max_digits_full_tkn', 'dist_min_digits_full_tkn', 'dist_rat_digits_full_tkn', \
  'dist_sum_digits_full_tkn', 'dist_dif_digits_full_tkn', 'dist_max_avg_len_tkn', 'dist_min_avg_len_tkn', 'dist_rat_avg_len_tkn', 'dist_sum_avg_len_tkn', \
  'dist_dif_avg_len_tkn', 'dist_max_unique_len_tkn', 'dist_min_unique_len_tkn', 'dist_rat_unique_len_tkn', 'dist_sum_unique_len_tkn', 'dist_dif_unique_len_tkn', \
  'dist_max_cap_tkn', 'dist_min_cap_tkn', 'dist_rat_cap_tkn', 'dist_sum_cap_tkn', 'dist_dif_cap_tkn', 'dist_max_digits_smpl', 'dist_min_digits_smpl', \
  'dist_rat_digits_smpl', 'dist_sum_digits_smpl', 'dist_dif_digits_smpl', 'dist_max_cap_big_smpl', 'dist_min_cap_big_smpl', 'dist_rat_cap_big_smpl', \
  'dist_sum_cap_big_smpl', 'dist_dif_cap_big_smpl', 'dist_max_unic_smpl', 'dist_min_unic_smpl', 'dist_rat_unic_smpl', 'dist_sum_unic_smpl', \
  'dist_dif_unic_smpl', 'dist_max_len_smpl', 'dist_min_len_smpl', 'dist_rat_len_smpl', 'dist_sum_len_smpl', 'dist_dif_len_smpl', \
  'dist_max_cap_smpl', 'dist_min_cap_smpl', 'dist_rat_cap_smpl', 'dist_sum_cap_smpl', 'dist_dif_cap_smpl']

CHAR_FEATURES =  ['_cv_cnt_l_3_q1', '_cv_cnt_l_3_q2', '_cv_cnt_lu_3_q1', '_cv_cnt_lu_3_q2', '_cv_jack_l_3', '_cv_cnt_l_2_q1', '_cv_cnt_l_2_q2', \
  '_cv_cnt_lu_2_q1', '_cv_cnt_lu_2_q2', '_cv_jack_l_2', '_cv_cnt_l_1_q1', '_cv_cnt_l_1_q2', '_cv_cnt_lu_1_q1', '_cv_cnt_lu_1_q2', '_cv_jack_l_1', \
  '_tf_cnt_l_3_q1', '_tf_cnt_l_3_q2', '_tf_cnt_lu_3_q1', '_tf_cnt_lu_3_q2', '_tf_jack_l_3', '_tf_cnt_l_2_q1', '_tf_cnt_l_2_q2', '_tf_cnt_lu_2_q1', \
  '_tf_cnt_lu_2_q2', '_tf_jack_l_2', '_tf_cnt_l_1_q1', '_tf_cnt_l_1_q2', '_tf_cnt_lu_1_q1', '_tf_cnt_lu_1_q2', '_tf_jack_l_1', '_tfidf_cnt_l_1_q1', \
  '_tfidf_cnt_l_1_q2', '_tfidf_cnt_lu_1_q1', '_tfidf_cnt_lu_1_q2', '_tfidf_jack_l_1', '_tfidf_cnt_l_2_q1', '_tfidf_cnt_l_2_q2', '_tfidf_cnt_lu_2_q1', \
  '_tfidf_cnt_lu_2_q2', '_tfidf_jack_l_2', '_tfidf_cnt_l_3_q1', '_tfidf_cnt_l_3_q2', '_tfidf_cnt_lu_3_q1', '_tfidf_cnt_lu_3_q2', '_tfidf_jack_l_3' ]

TOKEN_FEATURES = ['_cv_t_cnt_l_3_q1', '_cv_t_cnt_l_3_q2', '_cv_t_cnt_lu_3_q1', '_cv_t_cnt_lu_3_q2', '_cv_t_jack_l_3', '_cv_t_cnt_l_2_q1', '_cv_t_cnt_l_2_q2', \
  '_cv_t_cnt_lu_2_q1', '_cv_t_cnt_lu_2_q2', '_cv_t_jack_l_2', '_cv_t_cnt_l_1_q1', '_cv_t_cnt_l_1_q2', '_cv_t_cnt_lu_1_q1', '_cv_t_cnt_lu_1_q2', '_cv_t_jack_l_1', \
  '_tf_t_cnt_l_3_q1', '_tf_t_cnt_l_3_q2', '_tf_t_cnt_lu_3_q1', '_tf_t_cnt_lu_3_q2', '_tf_t_jack_l_3', '_tf_t_cnt_l_2_q1', '_tf_t_cnt_l_2_q2', '_tf_t_cnt_lu_2_q1', \
  '_tf_t_cnt_lu_2_q2', '_tf_t_jack_l_2', '_tf_t_cnt_l_1_q1', '_tf_t_cnt_l_1_q2', '_tf_t_cnt_lu_1_q1', '_tf_t_cnt_lu_1_q2', '_tf_t_jack_l_1', '_tfidf_t_cnt_l_1_q1', \
  '_tfidf_t_cnt_l_1_q2', '_tfidf_t_cnt_lu_1_q1', '_tfidf_t_cnt_lu_1_q2', '_tfidf_t_jack_l_1', '_tfidf_t_cnt_l_2_q1', '_tfidf_t_cnt_l_2_q2', '_tfidf_t_cnt_lu_2_q1', \
  '_tfidf_t_cnt_lu_2_q2', '_tfidf_t_jack_l_2', '_tfidf_t_cnt_l_3_q1', '_tfidf_t_cnt_l_3_q2', '_tfidf_t_cnt_lu_3_q1', '_tfidf_t_cnt_lu_3_q2', '_tfidf_t_jack_l_3' ]

FEATURES = SIMPLE_FEATURES1 + DIST_FEATURES1 + DIST_FEATURES2 + CHAR_FEATURES + TOKEN_FEATURES

USE_SVD = False
if USE_SVD:
  FEATURES.append('tfifd_svd_cosine')
  for i in range(20):
    FEATURES.append("svd_feature_{0}".format(i))
  print(FEATURES)
PATH_XGB_MODEL = '../data/xgb.pickle'

K_FOLDS = 5
EARLY_STOPS = 50
N_ESTIMATORS = 400
XGB_PARAMS = {
  'objective': 'binary:logistic'
  , 'eval_metric': 'logloss'
  , 'eta': 0.02 #0.11
  , 'max_depth': 4 #5
  , 'silent': 1
  #, 'learning_rate': 0.1
  #, 'n_estimators': 1000
  #, 'min_child_weight': 1
  #, 'gamma': 0.1
  #, 'subsample': 0.8
  #, 'seed': 27
  #, 'nthread': 4
  #, 'scale_pos_weight': 0.165
}
POSITIVE_RATE = 0.211


def parse_args():
  parser = argparse.ArgumentParser(description='xgb argparser')
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='submit preparation')  
  return parser.parse_args()

def describe_importance(bst, features):
  res = [(features[int(key[1:])], value) for key, value in bst.get_fscore().items()]
  return sorted(res, key = lambda x: -x[1])

def rescale(data):
  pos = data[data.is_duplicate == 1]
  neg = data[data.is_duplicate == 0]
  pos_length, neg_length = pos.shape[0], neg.shape[0]
  ratio = float(pos_length) / (pos_length + neg_length) 
  #scale = 1.*(1-POSITIVE_RATE-ratio)*(pos_length + neg_length)/neg_length/(1+POSITIVE_RATE)
  scale = (ratio-POSITIVE_RATE)*(pos_length+neg_length)/POSITIVE_RATE/neg_length
  while scale > 1:
    neg = pd.concat([neg, neg])
    scale -= 1
  n = int(scale * neg_length)
  neg = pd.concat([neg, neg.sample(n)])
  res = pd.concat([neg, pos])
  return res.sample(frac=1).reset_index(drop=True)

def cross_validation(data, features):
  print("cross valudation started...")
  #cv = CrossVal(data, K_FOLDS)
  xgb_data = xgb.DMatrix(data[features].values, label=data['is_duplicate'].values)
  #return xgb.cv(XGB_PARAMS, xgb_data, N_ESTIMATORS, folds=cv, metrics={'logloss'}, seed = 0)
  return xgb.cv(XGB_PARAMS, xgb_data, N_ESTIMATORS, nfold=K_FOLDS, metrics={'logloss'}, seed = 0)

def tuning_step1(data, features):
  param_test1 = {'max_depth': range(3,10,2), 'min_child_weight': range(1,6,2)}

  bst = xgb.XGBClassifier(learning_rate =0.1, 
    n_estimators=150,
    max_depth=5, 
    min_child_weight=1, 
    gamma=0, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=27)

  gsearch = GridSearchCV(estimator = bst, param_grid = param_test1, scoring='neg_log_loss', n_jobs=16, iid=False, cv=K_FOLDS)
  gsearch.fit(data[features].values, data.is_duplicate.values)
  print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)


def train_model(data, val = False, dump = False, features = FEATURES):
  bst = None
  if val:
    x_train, x_valid, y_train, y_valid = train_test_split(data[features].values, data['is_duplicate'].values, test_size=0.2, random_state=4242)
    xgb_data_train = xgb.DMatrix(x_train, label=y_train)
    xgb_data_valid = xgb.DMatrix(x_valid, label=y_valid)
    bst = xgb.train(XGB_PARAMS, xgb_data_train, N_ESTIMATORS, [(xgb_data_train, 'train'), (xgb_data_valid, 'valid')], early_stopping_rounds=50, verbose_eval=10)
  else:
    xgb_data_train = xgb.DMatrix(data[features].values, label=data['is_duplicate'].values)
    bst = xgb.train(XGB_PARAMS, xgb_data_train, N_ESTIMATORS, [(xgb_data_train, 'train')], early_stopping_rounds=50, verbose_eval=10)
  print(describe_importance(bst, features))
  if dump:
    pickle.dump(bst, open(PATH_XGB_MODEL, 'wb'))

def predict(data, features = FEATURES):
  bst = pickle.load(open(PATH_XGB_MODEL, "rb"))
  res = pd.DataFrame()
  xgb_data = xgb.DMatrix(data[features].values)
  predictions = bst.predict(xgb_data)
  res['test_id'] = data['test_id']
  res['is_duplicate'] = predictions
  res.to_csv('../data/v0.csv', index=False)

def main():
  args = parse_args()
  data = None
  xgb_data = None
  train = True
  if args.fin:
    train = False

  if train:
    data_train = load_data(True, False)
    data = rescale(data_train)
    columns = [col for col in data.columns if col not in ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'question1_tk', 'question2_tk']]  
    columns = [col for col in columns if col not in FEATURES_CORR]
    columns = [col for col in columns if col not in FEATURES_NOT_SELECTED]
    #print(cross_validation(data, columns)[-1:])
    #train_model(data, False, True, columns)
    tuning_step1(data, columns)
  else:
    print("start loading data")
    data = load_data(train, False) #train
    columns = [col for col in data.columns if col not in ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'question1_tk', 'question2_tk']]  
    columns = [col for col in columns if col not in FEATURES_CORR]
    columns = [col for col in columns if col not in FEATURES_NOT_SELECTED]
    predict(data, columns)
  
  #print(cross_validation(xgb_data)[-1:])
  

if __name__ == '__main__':
  main()