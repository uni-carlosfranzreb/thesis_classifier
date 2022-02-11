""" Initialize a training procedure, with several models, optimizers, and other
parameters available. """


import logging

import torch.optim

from hierarchy_mask import create_mask
from cnn.load_data import Dataset
from cnn.train import ModelTrainer


def init(params):
  """ Configure logging, log the parameters of this training procedure and
  initialize training. """
  log_params(params)
  dataset = Dataset(params["docs_folder"], params["subjects_file"],
      params["n_words"], params["n_dims"], params["shuffle"])
  logging.info(f'Dataset has {len(dataset)} documents')
  logging.info(f'There are {len(dataset.subjects)} subjects.\n')
  model = init_model(len(dataset.subjects), params)
  optimizer = init_optimizer(model, params)
  scheduler = init_scheduler(optimizer, params)
  trainer = ModelTrainer(params["run_id"], model, dataset)
  trainer.train(params["loss"], params["batch_size"], params["n_epochs"],
      optimizer, scheduler)


def init_model(n_subjects, params):
  """ Initialize the model. It can be ConvClassifier, SumClassifier or
  HierarchyClassifier """
  if 'hierarchy' in str(params["model"]):
    hierarchy_mask = create_mask(params["subjects_file"])
    return params["model"](params["n_dims"], hierarchy_mask, params["dropout"])
  elif params["dropout"] is not None:
    return params["model"](n_subjects, params["n_dims"], params["input_linear"],
        params["hidden_layer"], params["dropout"])
  return params["model"](n_subjects, params["n_dims"], params["input_linear"],
        params["hidden_layer"])


def init_optimizer(model, params):
  """ Initialize the optimizer. It can be 'SGD' or 'Adam'. """
  if params["optimizer"] == 'SGD':
    return torch.optim.SGD(model.parameters(), lr=params["lr"],
        momentum=params["momentum"], nesterov=True)
  elif params["optimizer"] == 'Adam':
    return torch.optim.Adam(model.parameters(), lr=params["lr"])
  else:
    raise ValueError('Optimizer is not supported.')


def init_scheduler(optimizer, params):
  """ Initialize the scheduler. Options are StepLR and OneCycleLR. """
  if params["scheduler"] is not None:
    if 'StepLR' in str(params["scheduler"]):
      return params["scheduler"](
        optimizer, step_size=params["scheduler_steps"],
        gamma=params["scheduler_gamma"]
      )      
    else:  # OneCycleLR
      return params["scheduler"](
        optimizer, params["lr"], total_steps=params["scheduler_steps"]
      )
  return None


def log_params(params):
  """ Log the model and training parameters. """
  logging.info(f'Training Run ID: {params["run_id"]}')
  logging.info(f'Classifier model: {params["model"]}')
  logging.info(f'Hidden layer size: {params["hidden_layer"]}')
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
  logging.info(f'Scheduler steps: {params["scheduler_steps"]}')
  logging.info(f'Scheduler gamma: {params["scheduler_gamma"]}')
  logging.info(f'Data shuffling?: {params["shuffle"]}\n')