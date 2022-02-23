""" Implementation of the coherent HMC model. It is exactly the same as the
convolutional model, but with an added layer on top.
"""


import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
  def __init__(self, n_labels, n_dims, mask, input_linear=10000,
      hidden_layer=1024, dropout=.1):
    """ Initializes the model.
    n_labels (int): no. of subjects in the classification problem.
    n_dims (int): no. of dimensions of each input word.
    input_linear (int): in_features of the first fully connected layer. Its
      default value corresponds to 400 words. For 250 words, it should be 6,200
    mask (tensor of shape (n_labels x n_labels)): a 1 means that the column
      subject is a descendant of the row subject. Diagonal is filled with ones.
      See descendant_mask.py for more details.
    """
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv1d(n_dims, 200, 5, padding='same')
    self.conv2 = nn.Conv1d(200, 100, 3, padding='same')
    self.dropout = nn.Dropout(dropout)
    self.fc1 = nn.Linear(input_linear, hidden_layer)
    self.fc2 = nn.Linear(hidden_layer, n_labels)
    self.mask = mask

  def forward(self, x):
    """ Computes the forward pass.
    x (tensor of size (batch_size x n_words x n_dims)): input.
    """
    x = F.max_pool1d(F.relu(self.conv1(x.transpose(1, 2))), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = self.dropout(x)
    x = F.relu(self.fc1(x.transpose(1, 2).flatten(start_dim=1)))
    x = torch.sigmoid(self.fc2(x))
    return self.mcm(x)
  
  def mcm(self, x):
    """Max-constraint module: the output for each subject must be at least as
    high as the outputs of its descendants. """
    masked = x.mul(self.mask.unsqueeze(1)).transpose(0, 1)
    return masked.max(dim=2).values
