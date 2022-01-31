""" Load the documents, and retrieve their respective word vectors from the
fasttext file. This class is an argument for PyTorch's DataLoader."""


from os import listdir
import json
import logging

import torch
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
  def __init__(self, folder, subjects_file):
    """ Initializes the Dataset.
    folder (str): location of the folder that contains the training files. The
      file structure is explained in the README.
    subjects_file (str): contains ID, name, etc. of subjects. They are
      mapped to the output neurons by their indices. """
    self.folder = folder
    self.subject_info = json.load(open(subjects_file))
    self.subjects = [subject_id for subject_id in self.subject_info]
    self.n_docs = 0
    for file in listdir(self.folder):
      docs = json.load(open(f'{self.folder}/{file}', encoding='utf-8'))
      self.n_docs += sum([len(v) for v in docs.values()])
  
  def __len__(self):
    """ Return the number of documents across all files. """
    return self.n_docs

  def __iter__(self):
    """ Iterate over files and return documents-subjects tuples. Files with
    'test' in the name are ignored. The subjects are an array with ones
    for assigned subjects and zeros elswhere. Both arrays are returned as
    tensors. """
    for file in listdir(self.folder):
      if 'test' in file:
        continue
      docs = json.load(open(file, encoding='utf-8'))
      for subject_id in docs:
        logging.info(f'Retrieving docs with subject {subject_id}')
        for doc in docs[subject_id]:
          subjects = torch.zeros(len(self.subjects))
          for subject in doc['subjects']:
            subjects[self.subjects.index(subject)] = 1
          return (torch.tensor(doc['data']), torch.tensor(subjects))
  
  def test_set(self, fname='test'):
    """ Load the test set into a dictionary and return it. If the test file is
    not called 'test', pass the name to the function.  """
    return json.load(open(f'{self.folder}{fname}.json'))