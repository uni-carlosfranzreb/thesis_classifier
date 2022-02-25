""" For each document of the repositories, compute the assignment probability
of each subject for the given model. The data file already contains the
fasttext vectors for the texts (see vectorize_repos in process_docs.py """


import json
import torch
from os import listdir

from cnn.convolutional_model import ConvClassifier
from cnn.coherent_model import CoherentClassifier

from descendant_mask import create_mask


def compute(model_file, dump_file, n_words=400, n_dims=300, hidden_size=100):
  data_folder = 'data/pretrained_vecs/data'
  subjects = list(json.load(open('data/openalex/subjects.json')).keys())
  input_linear = -1
  if n_words == 400:
    input_linear = 10000
  elif n_words == 250:
    input_linear = 6200
  try:
    model = ConvClassifier(len(subjects), n_dims, input_linear, hidden_size)
    model.load_state_dict(
      torch.load(model_file, map_location=torch.device('cpu'))
    )
  except RuntimeError:
    mask = create_mask('data/openalex/subjects.json')
    model = CoherentClassifier(len(subjects), n_dims, mask, input_linear,
      hidden_size)
    model.load_state_dict(
      torch.load(model_file, map_location=torch.device('cpu'))
    )
  model.eval()  # deactivate dropout
  assigned = {}
  for file in listdir(data_folder):
    data = json.load(open(f'{data_folder}/{file}'))
    for doc, vecs in data.items():
      assigned[doc] = {}
      data = prepare_data(vecs, n_words, n_dims)
      if data is None:
        continue
      with torch.no_grad():
        out = model(data).squeeze()
      for i in range(len(subjects)):
        assigned[doc][subjects[i]] = out[i].item()
  json.dump(assigned, open(dump_file, 'w'))


def prepare_data(data, n_words, n_dims):
  """ Resize vectors to fit the number of words, as done in load_data.py. """
  all_data = torch.tensor(data)
  if all_data.numel() == 0:
    return None
  if all_data.shape[0] >= n_words:
    data = all_data[:n_words]
  else:
    data = torch.zeros(n_words, n_dims)
    data[:all_data.shape[0]] = all_data
  return data.unsqueeze(0)
  

if __name__ == '__main__':
  runs = [(1645720728, 20), (1645707013, 20), (1645623620, 20), (1645527125, 20)]
  for run_id, epoch in runs:
    model_file = f'data/classifiers/{run_id}/epoch_{epoch}.pt'
    dump_file = f'data/classifiers/{run_id}/probabilities.json'
    compute(model_file, dump_file, n_words=250, hidden_size=100)