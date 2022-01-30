""" Process the file 'relevant_data.json', created by running the script
'retrieve_relevant_data.py' of the 'repository_analysis' repo. The processing
procedure is the same as for the vocabulary, to enable the comparison among
both sources. """


import json
from os import listdir

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


if __name__ == '__main__':
  process_docs()
