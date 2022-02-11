""" Run this script in the server to investigate an error. """

from os import listdir
import json

import torch

folder = 'data/openalex/split_filtered'
for file in listdir(folder):
  if 'test' in file:
    continue
  docs = json.load(open(f'{folder}/{file}', encoding='utf-8'))
  print(docs.keys())
  for doc in docs:
    print(doc)
    if doc['data'] == []:  # doc has no words
      continue
    all_data = torch.tensor(doc['data'])
    if all_data.shape[0] >= 250:
      data = all_data[:250]
    else:
      data = torch.zeros(250, 300)
      data[:all_data.shape[0]] = all_data