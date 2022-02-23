""" Create a torch mask that connects subjects with their descendants. It has
as many rows and columns as there are subjects. If the cell in the n-th row and
m-th column has a 1, it means that the m-th subject is a descendant of the n-th
subject. Thus, rows represent ancestors and columns, descendants. The diagonal
is also filled with ones, so that the output of each subject is also considered
when applying the MCM.

for all i: mask(i, i) = 1
for all i != j: mask(i, j) = 1 iff subject i is an ancestor of subject j
"""


import json
import torch


def create_mask(subjects_file):
  subjects = json.load(open(subjects_file))
  subject_ids = list(subjects.keys())
  mask = torch.eye(len(subjects), len(subjects), dtype=torch.uint8)
  for idx, data in enumerate(subjects.values()):
    for ancestor in data['ancestors']:
      if ancestor['id'] in subject_ids:
        mask[subject_ids.index(ancestor['id']), idx] = 1
  return mask


def test_mask():
  test_subjects = 'data/openalex/test_subjects.json'
  mask = create_mask(test_subjects)
  res = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1],  # Computer science
    [0, 1, 1, 1, 1, 1, 1],  # Machine learning
    [0, 0, 1, 0, 1, 0, 0],  # Clustering algorithms
    [0, 0, 0, 1, 0, 1, 1],  # Regression algorithms
    [0, 0, 0, 0, 1, 0, 0],  # K-nearest neighbors
    [0, 0, 0, 0, 0, 1, 1],  # Linear regression
    [0, 0, 0, 0, 0, 0, 1],  # Multiple linear regression
  ], dtype=torch.uint8)
  assert torch.all(mask == res)


if __name__ == '__main__':
  test_mask()
