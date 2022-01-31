""" Gargiulo's training procedure has the following hyperparameters:
1. batch size 10,
2. learning rate 0.1,
3. momentum 0.5,
4. uses Nesterov accelerated gradients

He does not mention clipping in the paper, so we will first try to train
the model without.
"""


import logging

import torch
from torch.utils.data import DataLoader

from classifier.load_data import Dataset
from classifier.model import Classifier


class ModelTrainer:
  def __init__(self, run_id, model, dataset):
    """ Initialize the trainer. 
    run_id (int): ID of this training run; used to save models.
    model (torch.nn): model to be trained.
    dataset (torch's Dataset): dataset to be used. """
    self.run_id = run_id
    self.model = model
    self.dataset = dataset
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.dump_folder = f'classifier/trained_models/{self.run_id}'

  def train(self, loss_fn, batch_size, n_epochs, lr, momentum):
    optimizer = torch.optim.SGD(
      self.model.parameters(), lr=lr, momentum=momentum, nesterov=True
    )
    for epoch in range(1, n_epochs+1):
      loader = DataLoader(self.dataset, batch_size=batch_size)
      self.cnt, self.current_loss = 0, 0  # for last 100 batches
      self.epoch_cnt, self.epoch_loss = 0, 0  # for epoch
      logging.info(f'Starting epoch {epoch}')
      for batch, labels in loader:
        optimizer.zero_grad()
        out = self.model(*[t.to(self.device) for t in batch])
        loss = loss_fn.backward(out, labels)
        optimizer.step()
        self.cnt += 1
        self.current_loss += loss
        if self.cnt % 100 == 0:
          self.log_loss()
      self.log_loss(epoch=epoch)
      self.evaluate()
  
  def evaluate(self):
    """ Evaluate the accuracy of the model with the test set. """
    test_set = self.dataset.test_set()
    self.model.eval()
    losses = []
    for doc in test_set:
      losses.append(self.model(doc['data']))
    logging.info(f'Avg. testing loss: {sum(losses)/len(losses)}')
    self.model.train()
  
  def log_loss(self, epoch=-1):
    """ If epoch=-1: log avg. loss of the last 100 batches. Before resetting
    the cnt and current_loss, add them to the totals for the epoch.
    Else: epoch has ended - log its avg. loss, set all counters to zero
    and call save_model(). """
    self.epoch_loss -= self.current_loss
    self.epoch_cnt += self.cnt
    if epoch > 0:
      avg_loss = self.epoch_loss / self.epoch_cnt
      logging.info(f'Avg. loss of epoch {epoch}: {avg_loss}')
      self.epoch_loss = 0
      self.epoch_cnt = 0
      self.save_model(epoch)
    else:
      avg_loss = self.current_loss / self.cnt
      logging.info(f'Avg. loss in the last 100 batches: {avg_loss}')
    self.cnt = 0
    self.current_loss = 0
  
  def save_model(self, epoch):
    """ Save the the model. The file should be named 'epoch_{epoch}', in the
    'run_id' folder. """
    torch.save(self.model.state_dict(), f'{self.dump_folder}/epoch_{epoch}.pt')
      

def init_training(run_id, docs_folder, subjects_file, n_words=400, n_dims=300,
    loss=torch.nn.BCELoss, batch_size=10, n_epochs=10, lr=.1):
  """ Configure logging, log the parameters of this training procedure and
  initialize training. """
  logging.info(f'Training Run ID: {run_id}')
  logging.info('Training classifier with the following parameters:')
  logging.info(f'Folder with documents: {docs_folder}')
  logging.info(f'File with subject information: {subjects_file}')
  logging.info(f'No. of words kept per document: {n_words}')
  logging.info(f'No. of dimensions per word: {n_dims}')
  logging.info(f'Training loss function: {loss}')
  logging.info(f'Batch size: {batch_size}')
  logging.info(f'No. of epochs: {n_epochs}')
  logging.info(f'Learning rate: {lr}\n')
  dataset = Dataset(docs_folder, subjects_file)
  logging.info(f'Dataset has {dataset.vocab.n_words} documents')
  logging.info(f'There are {len(dataset.subjects)} subjects.\n')
  model = Classifier(n_words, n_dims)
  trainer = ModelTrainer(run_id, model, dataset)
  trainer.train(loss, batch_size, n_epochs, lr)