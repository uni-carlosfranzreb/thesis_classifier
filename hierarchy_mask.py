""" Create a torch mask that connects MAG fields with their descendants. """


import json
import torch


def create_mask(subjects_file):
  subjects = json.load(open(subjects_file))
  fields = [id for id, data in subjects.items() if data['level'] == 0]
  mask = torch.zeros(len(fields), len(subjects)-len(fields), dtype=torch.uint8)
  for idx, data in enumerate(subjects.values()):
    if data['level'] == 0:
      continue
    subject_idx = idx - len(fields)  # assuming fields are before subjects
    for ancestor in data['ancestors']:
      if ancestor['id'] in fields:
        field_idx = fields.index(ancestor['id'])
        mask[field_idx, subject_idx] = 1
  return mask