import spacy
import numpy as np
import pandas as pd
nlp = spacy.load("en")
from utils import load_data, save_data


def sentence_similarity(sent1, sent2):
  doc1 = nlp(sent1)
  doc2 = nlp(sent2)
  return doc1.similarity(doc2)

def main():
  train = True
  data = load_data(train, False)
  data["spacy_sim"] = data.apply(lambda row: sentence_similarity(row["question1"], row["question2"]), axis = 1)
  save_data(data, train)

if __name__=="__main__":
  main()
