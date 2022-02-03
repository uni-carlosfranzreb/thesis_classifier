""" Train the classifier neural network. """


from time import time
import logging

from torch.nn import BCELoss

from classifier.train import init_training
from classifier.loss import AsymmetricLossOptimized


if __name__ == '__main__':
  run_id = int(time())
  docs_folder = 'data/openalex/split_docs'
  subjects_file = 'data/openalex/subjects.json'
  n_words = 400
  n_dims = 300
  loss = AsymmetricLossOptimized()
  batch_size = 10
  n_epochs = 10
  lr = .1
  momentum = .5
  logging.basicConfig(
    level=logging.INFO,
    filename=f'logs/training_{run_id}.log'
  )
  init_training(run_id, docs_folder, subjects_file, n_words, n_dims, loss,
    batch_size, n_epochs, lr, momentum
  )
