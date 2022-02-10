""" Test the classifier neural network. """


import torch

from cnn.model import Classifier


def test_forward(n_subjects, batch_size, n_words, n_dims):
  x = torch.rand((batch_size, n_words, n_dims))
  model = Classifier(n_subjects, n_dims)
  model(x)


if __name__ == '__main__':
  n_subjects = 1200
  batch_size=10
  n_words = 400
  n_dims = 300
  test_forward(n_subjects, batch_size, n_words, n_dims)