""" Test the classifier neural network. """


import torch

from cnn.convolutional_model import Classifier
from cnn.asymmetric_loss import ASL


def test_forward():
  n_subjects = 1200
  batch_size=10
  n_words = 400
  n_dims = 300
  x = torch.rand((batch_size, n_words, n_dims))
  model = Classifier(n_subjects, n_dims)
  model(x)


def test_asl():
  x = torch.tensor([.2, .4, .8])
  y = torch.tensor([0, 1, 0])
  bce = ASL(0, 0)
  focal_loss = ASL(1, 1)
  asl = ASL(2, 1)
  print(bce(x, y))  # should be 0.9
  print(focal_loss(x, y))  # should be 0.62
  print(asl(x, y))  # should be 0.52
  

if __name__ == '__main__':
  test_asl()