""" Train the classifier neural network. """


from time import time
import logging

from torch.nn import BCELoss
from torch.optim import lr_scheduler

from classifier.train import init_training
from classifier.loss import AsymmetricLossOptimized
from classifier.convolutional_model import ConvClassifier
from classifier.sum_model import SumClassifier



if __name__ == '__main__':
  """ Set the parameters for the training run. Optimizer can be 'SGD' or
  'Adam'. """
  params = {
    "run_id": int(time()),
    "model": SumClassifier,
    "docs_folder": 'data/openalex/split_docs',
    "subjects_file": 'data/openalex/subjects.json',
    "n_words": 400,
    "n_dims": 300,
    "dropout": .05,
    "loss": AsymmetricLossOptimized(),
    "batch_size": 10,
    "n_epochs": 10,
    "lr": .1,
    "momentum": None,
    "optimizer": 'Adam',
    "scheduler": lr_scheduler.OneCycleLR,
    "scheduler_steps": 3000,
    "shuffle": True,
    "hidden_layer": 200
  }
  logging.basicConfig(
    level=logging.INFO,
    filename=f'logs/training_{params["run_id"]}.log'
  )
  init_training(params)
