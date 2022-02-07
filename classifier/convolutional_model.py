""" Gargiulo's classifier. Its layers are as follows: 
1. convolutional layer with 200 filters, kernel size 5 and stride 1,
2. max-pooling layer with pool size 1,
3. convolutional layer with 100 filters, kernel size 3 and stride 1,
4. max-pooling layer with pool size 2,
5. dropout unit with discard probability 0.1 %,
6. fully connected layer with 1024 neurons and ReLU activation function,
7. fully connected output layer with as many neurons as there are subjects and
  sigmoid activation function.
! Both convolutional layers are one-dimensional, have padding and use the
! ReLU activation function.
In PyTorch, conv. layers pad with zeros by default.
"""


import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
  def __init__(self, n_labels, n_dims, hidden_layer=1024, dropout=.001):
    """ Initializes the model.
    n_labels (int): no. of subjects in the classification problem.
    n_dims (int): no. of dimensions of each input word.
    """
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv1d(n_dims, 200, 5, padding='same')
    self.conv2 = nn.Conv1d(200, 100, 3, padding='same')
    self.dropout = nn.Dropout(dropout)
    self.fc1 = nn.Linear(10000, hidden_layer)
    self.fc2 = nn.Linear(hidden_layer, n_labels)

  def forward(self, x):
    """ Computes the forward pass.
    x (tensor of size (batch_size x n_words x n_dims)): input.
    """
    x = F.max_pool1d(F.relu(self.conv1(x.transpose(1, 2))), 2)
    x = F.max_pool1d(F.relu(self.conv2(x)), 2)
    x = self.dropout(x)
    x = F.relu(self.fc1(x.transpose(1, 2).flatten(start_dim=1)))
    return torch.sigmoid(self.fc2(x))
