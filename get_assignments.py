""" For each document of the repositories, compute the assignment probability
of each subject for the given model. The data file already contains the
fasttext vectors for the texts (see vectorize_repos in process_docs.py """


import json
import torch

from classifier.model import Classifier


def compute(model_file, dump_file, n_words=400, n_dims=300):
  data = json.load(open(
    'data/json/dim/all/data_fasttext.json', encoding='utf-8'
  ))
  subjects = list(json.load(open('data/openalex/subjects.json')).keys())
  model = Classifier(len(subjects), n_dims)
  model.load_state_dict(torch.load(model_file))
  assigned = {}
  for doc, vecs in data.items():
    assigned[doc] = {}
    data = prepare_data(vecs, n_words, n_dims)
    out = model(data)
    for i in range(len(subjects)):
      assigned[doc][subjects[i]] = out[i]
  json.dump(assigned, open(dump_file, 'w'))


def prepare_data(doc, n_words, n_dims):
  """ Resize vectors to fit the number of words, as done in load_data.py. """
  all_data = torch.tensor(doc['data'])
  if all_data.shape[0] >= n_words:
    data = all_data[:n_words]
  else:
    data = torch.zeros(n_words, n_dims)
    data[:all_data.shape[0]] = all_data
  return data
  

if __name__ == '__main__':
  gargiulo_model = 'data/classifiers/1643821400/epoch_10.pt'
  baruch_model = 'data/classifiers/1643982969/epoch_2.pt'
  gargiulo_dump = 'data/classifiers/1643821400/probabilities.json'
  baruch_dump = 'data/classifiers/1643982969/probabilities.json'
  compute(gargiulo_model, gargiulo_dump)
  compute(baruch_model, baruch_dump)