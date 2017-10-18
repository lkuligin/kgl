import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import load_data, save_data, fit_min_max_scale

def mf1_merge(df1, df2):
  df1.columns = ["q", "set"]
  df2.columns = ["q", "set"]
  return pd.concat([df1, df2]).groupby("q")["set"].apply(lambda x: set.union(*x)).reset_index()

def mf1_groupby(data):
  q1 = data[["q1_idx", "q2_idx"]].groupby(["q1_idx"])["q2_idx"].apply(set).reset_index()
  q1.columns = ["q", "seq"]
  q2 = data[["q1_idx", "q2_idx"]].groupby(["q2_idx"])["q1_idx"].apply(set).reset_index()
  q2.columns = ["q", "seq"]
  return mf1_merge(q1, q2)

def magic1(data_train, data_test):
  tmp_train=mf1_groupby(data_train)
  tmp_test=mf1_groupby(data_test)
  tmp = mf1_merge(tmp_train, tmp_test)
  tmp["amount"] = tmp.seq.apply(lambda x: len(x))
  questions_occurence = tmp.set_index("q").T.squeeze().to_dict()
  
  m_train = data_train.apply(lambda row: questions_occurence[row["question1"]].intersect(questions_occurence[row["question1"]])).values
  m_test = data_test.apply(lambda row: questions_occurence[row["question1"]].intersect(questions_occurence[row["question1"]])).values

  m_train, m_test = fit_min_max_scale(m_train, m_test)
  data_tran["magic_1"] = m_train
  data_test["magic_1"] = m_test

def enumerate_questions(data_train, data_test):
  all_questions = pd.concat([data_train.question1.values, data_train.question2.values, data_test.question1.values, data_test.question2.values]).unique()
  all_q = pd.DataFrame(all_questions)
  all_q.columns = ["question"]
  all_q["index1"] = all_q.index
  enrich_q_with_index(data_train, all_q)
  enrich_q_with_index(data_test, all_q)
  save_data(data_train, True)
  save_data(data_test, False)

def enrich_q_with_index(data, questions)
  data = pd.merge(data, questions, left_on=["question1"], right_on=["question"])
  data.rename(inplace = True, columns={"index1":"q1_idx"})
  data = pd.merge(data, questions, left_on=["question2"], right_on=["question"])
  data.rename(inplace = True, columns={"index1":"q2_idx"})

def main():
  print("start")

if __name__ == "__main__":
  main()