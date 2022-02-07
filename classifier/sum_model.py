""" Similar to the convolutional model, but the convolutions are replaced
with vector addition. """


import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
  def __init__(self, n_labels, n_dims, hidden_layer=200):
    """ Initializes the model.
    n_labels (int): no. of subjects in the classification problem.
    n_dims (int): no. of dimensions of each input word.
    """
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(n_dims, hidden_layer)
    self.fc2 = nn.Linear(hidden_layer, n_labels)

  def forward(self, x):
    """ Computes the forward pass.
    x (tensor of size (batch_size x n_words x n_dims)): input.
    The word vectors are summed and then fed to the feed-forward NN. """
    x = F.relu(self.fc1(x.sum(1)))
    return torch.sigmoid(self.fc2(x))
