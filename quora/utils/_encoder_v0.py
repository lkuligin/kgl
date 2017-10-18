#!/Users/lkuligin/Documents/ml/py3/bin/python
# coding=utf-8

from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import load_data, save_data


STOP_SYMBOLS = ['\n']
REPLACE_SYMBOLS = []
STOPWORDS = []
SVD_COMPONENTS = 200
#STOPWORDS = stopwords.words('english')
PATH_COUNT_VECTORIZER = '../data/count_vectorizer.pickle'
PATH_TFIDF_VECTORIZER = '../data/tfidf_vectorizer.pickle'
PATH_COUNT_MATRIX = '../data/count_matrix.pickle'
PATH_TFIDF_MATRIX = '../data/tfidf_matrix.pickle'
PATH_SVD_MATRIX = '../data/svd_matrix.pickle'
PATH_SVD = '../data/svd_matrix.pickle'

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer('\w+')

def parse_args():
  #TODO add argument for test/train dataset
  parser = argparse.ArgumentParser(description='Creates vectorizer')
  parser.add_argument('-c', '--cnt', action='store_const', const=True, help='count vectorizer')
  parser.add_argument('-s', '--svd', action='store_const', const=True, help='svd transformation')
  parser.add_argument('-t', '--tf', action='store_const', const=True, help='tfidf vectorizer')
  parser.add_argument('-e', '--enr', action='store_const', const=True, help='enrich and pickle the train dataframe')
  parser.add_argument('-r', '--rec', action='store_const', const=True, help='recalculate and pickle')
  parser.add_argument('-l', '--let', action='store_const', const=True, help='letter transformation')
  
  return parser.parse_args()

def preprocess(line):
  line = re.sub(r"what's", "what is ", line)
  line = re.sub(r"\'ve", " have ", line)
  line = re.sub(r"can't", "cannot ", line)
  line = re.sub(r"n't", " not ", line)
  line = re.sub(r"i'm", "i am ", line)
  line = re.sub(r"\'re", " are ", line)
  line = re.sub(r"\'d", " would ", line)
  line = re.sub(r"\'ll", " will ", line)
  line = re.sub(r"(\d+)(k)", r"\g<1>000", line)
  line = re.sub(r" e g ", " eg ", line)
  line = re.sub(r" b g ", " bg ", line)
  line = re.sub(r" u s ", " american ", line)
  line = re.sub(r"\0s", "0", line)
  line = re.sub(r" 9 11 ", "911", line)
  line = re.sub(r"\'s", " ", line)
  line = re.sub(r",", ", ", line)
  line = re.sub(r"\.", ". ", line)
  line = re.sub(r"!", " ! ", line)
  line = re.sub(r"\/", " ", line)
  line = re.sub(r"\|", " | ", line)
  line = re.sub(r"\^", " ^ ", line)
  line = re.sub(r"\+", " + ", line)
  line = re.sub(r"\-", " - ", line)
  line = re.sub(r"\=", " = ", line)
  line = re.sub(r"'", " ", line)
  line = re.sub(r":", " : ", line)
  line = re.sub(r"e\s*-\s*mail", "email", line)
  line = re.sub(r"j k", "jk", line)
  return line

def parse_line(line):
  """Returns a tokenized string based on regexp parser
  args:
    line - string to tokenize
  return:
    array of tokens, stopwords excluded
  """
  return [stemmer.stem(word) for word in tokenizer.tokenize(line.lower()) if len(word)>0 and word not in STOPWORDS]

def get_count_vectorizer(lines):
  """Returns a count-occurences matrix 
  args:
    lines = sequence of lines
  return:
    matrix
    features' names
  """
  cv = CountVectorizer(input='content'
    , strip_accents = 'unicode'
    , analyzer = 'word'
    , tokenizer = parse_line
  )
  cv_matrix = cv.fit_transform(lines)
  return cv, cv_matrix

def get_tfidf_vectorizer(lines):
  tfv = TfidfVectorizer(input = 'content'
    , strip_accents = 'unicode' #'ascii' #None 
    , analyzer = 'word'
    , tokenizer = parse_line
    , use_idf = True
    , smooth_idf = True
    , sublinear_tf = False
    , norm = 'l2'
  )
  matrix = tfv.fit_transform(lines)
  return tfv, matrix

def cosine_distance(cv, line1, line2):
  v1, v2 = cv.transform([line1, line2]).toarray()
  dist = spatial.distance.cosine(v1, v2) if (sum(v1)>0 and sum(v2)>0) else 0
  return dist

def main():
  args = parse_args()
  train = True
  initial = False
  data = load_data(train, initial)

  cv = None
  tfv = None
  tfv_matrix = None
  all_questions = pd.concat([data.question1, data.question2]).unique()
  svd = None

  if args.rec:
    if args.cnt:
      cv, _ = get_count_vectorizer(all_questions)
      pickle.dump(cv, open(PATH_COUNT_VECTORIZER, 'wb'))
      print("count vect dumped and updated")
    if args.tf:
      tfv, tfv_matrix = get_tfidf_vectorizer(all_questions)
      pickle.dump(tfv, open(PATH_TFIDF_VECTORIZER, 'wb'))
      pickle.dump(tfv_matrix, open(PATH_TFIDF_MATRIX, 'wb'))
      print("tfidf vectorizer dumped and updated")
    if args.svd:
      if not tfv_matrix:
        print("preparing the matrix")
        _, tfv_matrix = get_tfidf_vectorizer(all_questions)
        pickle.dump(tfv_matrix, open(PATH_TFIDF_MATRIX, "wb" ))
        print("matrix prepared!")
        #svd = TruncatedSVD(n_components = SVD_COMPONENTS)
        #svd.fit_transform(tfv_matrix)
        #pickle.dump(svd, open(PATH_SVD_MATRIX, "wb" ))
  if args.enr:
    if args.cnt:
      if not args.rec:
        cv  = pickle.load(open(PATH_COUNT_VECTORIZER, "rb"))
      analyzer = cv.build_analyzer()
      data['question1_tk'] = data.question1.apply(lambda x: analyzer(x))
      data['question2_tk'] = data.question2.apply(lambda x: analyzer(x))
      m1 = cv.transform(data.question1.tolist())
      m2 = cv.transform(data.question2.tolist())
      norms1 = np.sqrt(m1.multiply(m1).sum(1))
      norms2 = np.sqrt(m2.multiply(m2).sum(1))
      cos = m1.multiply(m2).sum(1)
      data['simple_norm1']  = [el[0] for el in norms1.tolist()]
      data['simple_norm2']  = [el[0] for el in norms2.tolist()]
      data['simple_dot']  = [el[0] for el in cos.tolist()]
      save_data(data, train)
      print("data updated for count_vectorizer features")
    if args.tf:
      if not args.rec:
        tfv  = pickle.load(open(PATH_TFIDF_VECTORIZER, "rb"))
      m1 = tfv.transform(data.question1.tolist())
      m2 = tfv.transform(data.question2.tolist())
      cos = m1.multiply(m2).sum(1)
      #data['tfidf_norm1']  = [el[0] for el in norms1.tolist()]
      #data['tfidf_norm2']  = [el[0] for el in norms2.tolist()]
      data['tfidf_cosine']  = [el[0] for el in cos.tolist()]
      m1 = m1.ceil()
      m2 = m2.ceil()
      common = m1.multiply(m2)
      cweights = (m1.multiply(common) + m2.multiply(common)).sum(1)
      data['tfidf_cweights'] = [el[0] for el in cweights.tolist()]
      save_data(data, train)
      print("data updated for tfidf features")
    if args.svd:
      if not args.rec:
        tfv = pickle.load(open(PATH_TFIDF_VECTORIZER, "rb"))
        svd = pickle.load(open(PATH_SVD, "rb"))
      t1 = tfv.transform(data.question1.tolist())
      t2 = tfv.transform(data.question2.tolist())
      m1 = svd.transform(t1)
      m2 = svd.transform(t2)
      cos = np.multiply(m1, m2).sum(1)/np.sqrt(np.multiply(m1, m1).sum(1))/np.sqrt(np.multiply(m2, m2).sum(1))
      data['tfifd_svd_cosine'] = cos
      print("cosine sim calculated!")
      for i in range(20):
        feature_name = "svd_feature_{0}".format(i)
        f1 = m1[:, i]
        f2 = m2[:, i]
        data[feature_name] = [min(abs(el1), abs(el2))/max(abs(el1), abs(el2)) for el1, el2 in zip(f1, f2)]
        print(i) 
      save_data(data, train)


  #tfidf_matrix = get_tfidf_matrix(count_matrix)
  #svd = TruncatedSVD(n_components = SVD_COMPONENTS)
  #svd.fit_transform(tfidf_matrix)
  #pickle.dump(svd, open( "./data/svd.pickle", "wb" ))

if __name__ == '__main__':
  main()