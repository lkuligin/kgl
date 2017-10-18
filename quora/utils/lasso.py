import numpy as np
from scipy.stats.stats import pearsonr
import itertools
import pandas as pd
from sklearn.linear_model import RandomizedLasso
from sklearn.svm import LinearSVC

FEATURES_CORR = ['dist_dif_cap_big_tkn', 'dist_max_cap_big_tkn', '_tf_t_cnt_lu_2_q1', '_tfidf_cnt_lu_1_q2', '_cv_cnt_l_1_q1', \
'_tf_cnt_l_3_q2', '_tf_t_n2_sgd_intersect_', 'distance_cv_cosine', '_tfidf_t_jack_l_1', '_tf_t_jack_l_1', 'dist_sum_cap_big_smpl', '_tf_t_n3_nb_diff_', \
'_tf_cnt_lu_1_q1', 'distance_tf_canberra', '_tf_t_cnt_lu_3_q1', '_tf_jack_l_3', '_tfidf_t_cnt_lu_1_q2', '_tfidf_cnt_l_3_q2', '_svd_tf_n1_h_', \
'dist_rat_cap_tkn', '_tf_t_cnt_l_3_q2', '_tfidf_t_n2_sgd_diff_', '_tf_t_cnt_lu_1_q2', '_tfidf_cnt_lu_2_q2', 'dist_sum_len_tkn', '_tf_jack_l_2', \
'dist_min_digits_tkn', 'dist_max_len_tkn', 'dist_min_unique_len_tkn', '_cv_t_cnt_l_3_q1', 'distance_cv_l1', '_cv_jack_l_2', '_tf_cnt_lu_3_q1', \
'_tf_t_cnt_lu_2_q2', '_tfidf_t_cnt_l_2_q1', 'dist_rat_len_tkn', '_tfidf_cnt_l_3_q1', '_cv_t_cnt_l_2_q2', 'dist_min_cap_big_tkn', 'dist_min_len_tkn', \
'_tfidf_t_n1_nb_intersect_', '_tfidf_t_n3_nb_intersect_', '_tfidf_t_cnt_lu_2_q1', 'dist_min_cap_tkn', '_cv_t_n3_nb_diff_', '_tf_cnt_l_3_q1', \
'_tfidf_jack_l_3', '_tfidf_cnt_lu_2_q1', 'dist_rat_digits_tkn', '_cv_t_cnt_lu_2_q1', 'dist_sum_cap_big_tkn', '_tfidf_t_cnt_lu_1_q1', \
'_tfidf_t_cnt_lu_3_q2', '_tfidf_n2_sgd_diff_', '_tf_n1_sgd_diff_', '_cv_t_cnt_lu_2_q2', '_tf_t_n3_sgd_intersect_', '_tf_t_cnt_lu_3_q2', \
'_tfidf_t_cnt_l_2_q2', 'dist_dif_cap_big_smpl', '_cv_cnt_lu_2_q2', '_cv_t_cnt_l_2_q1', 'dist_rat_cap_big_smpl', '_tfidf_t_cnt_lu_3_q1', \
'_cv_t_cnt_lu_3_q2', 'dist_dif_digits_tkn', 'dist_max_cap_tkn', 'dist_sum_digits_tkn', 'distance_cv_jaccard', '_tfidf_t_cnt_lu_2_q2', \
'_cv_cnt_lu_3_q1', '_tfidf_cnt_lu_1_q1', 'dist_sum_cap_tkn', '_tf_t_cnt_l_3_q1', 'dist_dif_len_tkn', '_tf_cnt_lu_2_q2', '_tf_t_n3_sgd_diff_', \
'_tf_t_cnt_lu_1_q1', 'dist_dif_cap_tkn', 'dist_rat_cap_big_tkn', '_cv_cnt_l_2_q2', '_cv_t_cnt_lu_3_q1', '_tfidf_cnt_lu_3_q1', '_cv_cnt_l_2_q1', \
'_cv_t_cnt_l_3_q2', '_tf_t_cnt_l_2_q2', '_tf_cnt_lu_3_q2', '_cv_cnt_l_1_q2', '_tf_t_cnt_l_2_q1', 'distance_tf_jaccard', '_tfidf_t_cnt_l_3_q2', \
'_tf_t_n1_nb_diff_', '_cv_cnt_lu_3_q2', '_tf_cnt_lu_1_q2', '_tfidf_t_n2_nb_diff_', 'dist_max_digits_tkn', '_tf_cnt_lu_2_q1', '_cv_cnt_lu_2_q1', \
'_svd_dist_l2_tfidf_n3_', '_tfidf_t_cnt_l_3_q1', '_cv_t_jack_l_1', '_tfidf_cnt_lu_3_q2', 'dist_sum_unique_len_tkn']


FEATURES_NOT_SELECTED = ['dist_max_unic_smpl', 'dist_dif_unic_smpl', 'dist_min_unic_smpl', 'dist_rat_unic_smpl', 'dist_sum_unic_smpl', '_svd_dist_l1_tf_t_n1_', \
'_cv_cnt_l_3_q2', '_svd_dist_cos_tfidf_t_n3_', '_cv_cnt_l_3_q1', '_tfidf_cnt_l_1_q2', 'dist_sum_len_smpl', '_tf_cnt_l_2_q1', '_svd_dist_l1_tf_t_n2_']

def eliminate_correlations():
  columns = [col for col in data.columns if col not in ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'question1_tk', 'question2_tk']]
  correlations = {}
  columns_to_eliminate = set()
  for col_a, col_b in itertools.combinations(columns, 2):
    if col_b not in columns_to_eliminate:
      corr =  pearsonr(data.loc[:, col_a], data.loc[:, col_b]) 
      if abs(corr[0]) > 0.98:
        correlations[col_a + '__' + col_b] = corr 
        print(col_a, col_b, corr)
        columns_to_eliminate.add(col_b)

def lasso():
  columns = [col for col in data.columns if col not in ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'question1_tk', 'question2_tk']]  
  columns = [col for col in columns if col not in FEATURES_CORR]
  X = data[columns]
  X.fillna(0, inplace = True)
  Y = data.is_duplicate
  rlasso = RandomizedLasso(alpha=0.025)
  rlasso.fit(X, Y)
  print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), columns), reverse=True))

  svm = LinearSVC(C=0.75)
  svm.fit(X,Y)
  print(sorted(zip(map(lambda x: abs(round(x, 4)), svm.coef_[0]), columns), reverse=True))