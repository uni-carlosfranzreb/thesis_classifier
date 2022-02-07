""" Similar to the convolutional model, but the hidden layer has the size of
the number of fields (19), and the output layer nodes (i.e. the subjects) are
only connected to the fields they have as ancestors. """


import torch
from torch import nn
import torch.nn.functional as F

from classifier.masked_linear import MaskedLinear


class Classifier(nn.Module):
  def __init__(self, n_dims, mask, dropout=.001):
    """ Initializes the model.
    n_labels (int): no. of subjects in the classification problem.
    n_dims (int): no. of dimensions of each input word.
    """
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv1d(n_dims, 200, 5, padding='same')
    self.conv2 = nn.Conv1d(200, 100, 3, padding='same')
    self.dropout = nn.Dropout(dropout)
    self.fc1 = nn.Linear(10000, mask.shape[0])
    self.fc2 = MaskedLinear(mask, mask.shape[0], mask.shape[1])

  def forward(self, x):
    """ Computes the forward pass.
    x (tensor of size (batch_size x n_words x n_dims)): input.
    """
    x = F.max_pool1d(F.relu(self.conv1(x.transpose(1, 2))), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = self.dropout(x)
    x = F.relu(self.fc1(x.transpose(1, 2).flatten(start_dim=1)))
    return torch.sigmoid(self.fc2(x))
