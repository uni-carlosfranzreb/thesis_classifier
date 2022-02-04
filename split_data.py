""" Split the data into training and test sets. """


import json
from os import listdir
from random import sample
from math import ceil
import logging


def split(docs_folder, dump_folder, test_pctg=.01):
  """ Split the data into two files with the ts_file having test_pctg % of
  the data and tr_file having the rest. Split the data of each subject, so the
  test file contains samples with all subjects. To avoid sampling zero docs of
  a subject, ceil the result of n_docs * test_pctg. """
  ts_docs = []
  for file in listdir(docs_folder):
    docs = json.load(open(f'{docs_folder}/{file}', encoding='utf-8'))
    tr_docs = []
    for subject in docs:
      test_cnt = ceil(len(docs[subject]) * test_pctg)
      logging.info(f'{subject} test set: {test_cnt} docs')
      ts_docs += sample(docs[subject], test_cnt)
      tr_docs += [d for d in docs[subject] if d not in ts_docs]
    logging.info(f'Training set of {file}: {len(tr_docs)} docs')
    json.dump(tr_docs, open(f'{dump_folder}/{file}', 'w', encoding='utf-8'))
  json.dump(ts_docs, open(f'{dump_folder}/test.json', 'w', encoding='utf-8'))
  logging.info(f'Test has {len(ts_docs)} files')


def debug_data():
  """ Create smaller training and test files for debugging. """
  split_folder = 'data/openalex/split_docs'
  dump_folder = 'data/openalex/debug_docs'
  for file in listdir(split_folder):
    docs = json.load(open(f'{split_folder}/{file}'))
    json.dump(docs[:20], open(f'{dump_folder}/{file}', 'w'))


if __name__ == '__main__':
  # logging.basicConfig(
  #   level=logging.INFO,
  #   filename='logs/split_data.log',
  #   format='%(message)s'
  # )
  # split('data/openalex/doc_vecs', 'data/openalex/split_docs')
  debug_data()