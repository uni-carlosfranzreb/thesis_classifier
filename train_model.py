""" Train the classifier neural network. """


from time import time
import logging

from torch.nn import BCELoss
from torch.optim import lr_scheduler

from cnn.init_training import init
from cnn.asymmetric_loss import AsymmetricLossOptimized
from cnn.convolutional_model import Classifier as ConvClassifier
from cnn.sum_model import Classifier as SumClassifier
from cnn.hierarchy_model import Classifier as HierarchyClassifier


if __name__ == '__main__':
  """ Set the parameters for the training run.
  - Models: ConvClassifier, SumClassifier, HierarchyClassifier.
  - Losses: BCELoss, AsymmetricLossOptimized.
  - Scheduler can also be None.
  - Optimizer can be 'SGD' or 'Adam'. """
  params = {
    "run_id": int(time()),
    "model": ConvClassifier,
    "subjects_file": 'data/openalex/subjects.json',
    "docs_folder": 'data/openalex/split_docs',
    "subjects_file": 'data/openalex/subjects.json',
    "n_words": 250,
    "n_dims": 300,
    "dropout": .7,
    "loss": BCELoss(),
    "batch_size": 10,
    "n_epochs": 30,
    "lr": .4,
    "momentum": .5,
    "optimizer": 'SGD',
    "scheduler": lr_scheduler.StepLR,
    "scheduler_steps": 2,
    "scheduler_gamma": .9,
    "shuffle": True,
    "hidden_layer": 100,
    "input_linear": 6200
  }
  logging.basicConfig(
    level=logging.INFO,
    filename=f'logs/training_{params["run_id"]}.log'
  )
  init(params)
