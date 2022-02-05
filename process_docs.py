""" Process the publications retrieved from OpenAlex, as was done with the
documents of the repositories. """


import json
from os import listdir
import logging

from flair.data import Sentence
from flair.tokenization import SpacyTokenizer
from flair.models import SequenceTagger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class DataProcessor:
  def __init__(self, tokenizer, tagger, lemmatizer):
    self.tokenizer = tokenizer
    self.tagger = tagger
    self.lemmatizer = lemmatizer

  def process_docs(self, docs):
    """ Process documents. """
    processed = {}
    for subject in docs:
      processed[subject] = []
      for doc in docs[subject]:
        processed[subject].append({
          'data': self.process(Sentence(doc['data'])),
          'subjects': doc['subjects']
        })
    return processed

  def process(self, sentence):
    """ Given a Sentence object, lower-case and lemmatize the words. """
    self.tagger.predict(sentence)
    tag_dict = {
      'ADJ': wordnet.ADJ,
      'NOUN': wordnet.NOUN,
      'VERB': wordnet.VERB,
      'ADV': wordnet.ADV
    }
    lemmas = []
    for token in sentence:
      if token.labels[0].value in tag_dict:
        lemmas.append(self.lemmatizer.lemmatize(
          token.text.lower(), tag_dict[token.labels[0].value])
        )
      else:
        lemmas.append(token.text.lower())
    return lemmas


def process_docs():
  tokenizer = SpacyTokenizer('en_core_web_sm')
  lemmatizer = WordNetLemmatizer()
  tagger = SequenceTagger.load('upos-fast')
  processor = DataProcessor(tokenizer, tagger, lemmatizer)
  doc_folder = 'data/openalex/docs'
  dump_folder = 'data/openalex/processed_docs'
  for file in listdir(doc_folder):
    docs = json.load(open(f'{doc_folder}/{file}', encoding='utf-8'))
    processed = processor.process_docs(docs)
    json.dump(processed, open(f'{dump_folder}/{file}', 'w', encoding='utf-8'))


def get_vecs():
  """ Retrieve the vectors of the docs and dump them in another folder. """
  docs_folder = 'data/openalex/processed_docs'
  vecs_folder = 'data/openalex/doc_vecs'
  fname = 'data/pretrained_vecs/wiki-news-300d-1M-subword.vec'
  fin = open(fname, encoding='utf-8', newline='\n', errors='ignore')
  pretrained = {}
  fin.readline()  # skip first line
  for line in fin:
    tokens = line.rstrip().split(' ')
    pretrained[tokens[0]] = list(map(float, tokens[1:]))
  for file in listdir(docs_folder):
    docs = json.load(open(f'{docs_folder}/{file}', encoding='utf-8'))
    vecs = {}
    for subject in docs:
      vecs[subject] = []
      for doc in docs[subject]:
        vecs[subject].append({'data': [], 'subjects': doc['subjects']})
        for w in doc['data']:
          if w in pretrained:
            vecs[subject][-1]['data'].append(pretrained[w])
        found = len(vecs[subject][-1]["data"])
        logging.info(f'Found {found} vecs for {len(doc["data"])} words')
    json.dump(vecs, open(f'{vecs_folder}/{file}', 'w', encoding='utf-8'))


def find_vec(token):
  """ Return the vector for the given token or None if not found. """
  fname = 'data/pretrained_vecs/wiki-news-300d-1M-subword.vec'
  for line in open(fname, encoding='utf-8', newline='\n', errors='ignore'):
    if line[:len(token)+1] == token + ' ':
      tokens = line.rstrip().split(' ')
      return list(map(float, tokens[1:]))
  logging.info(f'{token} not found')


if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO, 
    handlers=[logging.FileHandler('logs/get_vecs.log', 'w', 'utf-8')],
    format='%(message)s'
  )
  get_vecs()
