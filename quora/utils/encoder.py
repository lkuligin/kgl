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
  parser.add_argument('-f', '--fin', action='store_const', const=True, help='submit preparation')
  parser.add_argument('-i', '--ini', action='store_const', const=True, help='initial data read')
  
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
  initial = False
  if args.ini:
    initial = True
  data_train = load_data(True, initial)
  data_test = load_data(False, initial)

  cv = None
  tfv = None
  tfv_matrix = None
  all_questions = pd.concat([data_train.question1, data_train.question2, data_test.question1, data_test.question2]).unique()
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
  if args.enr:
    if args.cnt:
      if not args.rec:
        cv  = pickle.load(open(PATH_COUNT_VECTORIZER, "rb"))
      analyzer = cv.build_analyzer()
      if not args.fin:
        data_train['question1_tk'] = data_train.question1.apply(lambda x: analyzer(x))
        data_train['question2_tk'] = data_train.question2.apply(lambda x: analyzer(x))
        save_data(data_train)
      else:
        data_test['question1_tk'] = data_test.question1.apply(lambda x: analyzer(x))
        data_test['question2_tk'] = data_test.question2.apply(lambda x: analyzer(x))
        save_data(data_test, False)
      print("data updated for count_vectorizer features")

if __name__ == '__main__':
  main()