""" Initialize a training procedure, with several models, optimizers, and other
parameters available. """


import logging

import torch

from classifier.load_data import Dataset
from classifier.train import ModelTrainer


def init_training(params):
  """ Configure logging, log the parameters of this training procedure and
  initialize training. """
  logging.info(f'Training Run ID: {params["run_id"]}')
  logging.info('Training classifier with the following parameters:')
  logging.info(f'Folder with documents: {params["docs_folder"]}')
  logging.info(f'File with subject information: {params["subjects_file"]}')
  logging.info(f'No. of words kept per document: {params["n_words"]}')
  logging.info(f'No. of dimensions per word: {params["n_dims"]}')
  logging.info(f'Dropout probability: {params["dropout"]}')
  logging.info(f'Training loss function: {params["loss"]}')
  logging.info(f'Optimizer: {params["optimizer"]}')
  logging.info(f'Batch size: {params["batch_size"]}')
  logging.info(f'No. of epochs: {params["n_epochs"]}')
  logging.info(f'Learning rate: {params["lr"]}')
  logging.info(f'Momentum: {params["momentum"]}')
  logging.info(f'Scheduler: {params["scheduler"]}')
  logging.info(f'Data shuffling?: {params["shuffle"]}\n')
  dataset = Dataset(params["docs_folder"], params["subjects_file"],
      params["n_words"], params["n_dims"], params["shuffle"])
  n_subjects = len(dataset.subjects)
  logging.info(f'Dataset has {len(dataset)} documents')
  logging.info(f'There are {len(dataset.subjects)} subjects.\n')
  if params["dropout"] is not None:
    model = params["model"](n_subjects, params["n_dims"], params["dropout"])
  else:
    model = params["model"](n_subjects, params["n_dims"])
  if params["optimizer"] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"],
        momentum=params["momentum"], nesterov=True)
  elif params["optimizer"] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
  else:
    raise ValueError('Optimizer is not supported.')
  if params["scheduler"] is not None:
    scheduler = params["scheduler"](optimizer, params["lr"], total_steps=2000)
  else:
    scheduler = None
  trainer = ModelTrainer(params["run_id"], model, dataset)
  trainer.train(params["loss"], params["batch_size"], params["n_epochs"],
      optimizer, scheduler)