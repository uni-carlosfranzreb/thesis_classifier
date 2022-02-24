""" Gargiulo's training procedure has the following hyperparameters:
1. batch size 10,
2. learning rate 0.1,
3. momentum 0.5,
4. uses Nesterov accelerated gradients

He does not mention clipping in the paper, so we will first try to train
the model without.

Ben-Baruch uses the asymmetric loss with learning rate 2e-4, the Adam optimizer
and 1-cycle scheduler.
"""


import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss


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
    self.dump_folder = f'models/{self.run_id}'
    if os.path.exists(self.dump_folder):
      raise ValueError(f'Folder {run_id} already exists. It should not')
    else:
      os.mkdir(self.dump_folder)

  def train(self, loss_fn, batch_size, n_epochs, optimizer, scheduler=None):
    loss_fn.to(self.device)
    for epoch in range(1, n_epochs+1):
      loader = DataLoader(self.dataset, batch_size=batch_size)
      self.cnt, self.current_loss = 0, 0  # for last 100 batches
      self.epoch_cnt, self.epoch_loss = 0, 0  # for epoch
      logging.info(f'Starting epoch {epoch}')
      for batch in loader:
        data, labels = [t.to(self.device) for t in batch]
        optimizer.zero_grad()
        out = self.model(data)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        self.cnt += 1
        self.current_loss += loss
        if self.cnt % 100 == 0:
          self.update_lr(optimizer, scheduler)
      self.update_lr(optimizer, scheduler, epoch=epoch)
      self.evaluate()
  
  def evaluate(self, loss_fn=BCELoss()):
    """ Evaluate the accuracy of the model with the test set. """
    self.model.eval()
    losses = []
    with torch.no_grad():
      for doc in self.dataset.test_set():
        data, labels = [t.unsqueeze(0).to(self.device) for t in doc]
        out = self.model(data)
        loss = loss_fn(out, labels)
        losses.append(loss)
    logging.info(f'Avg. testing loss: {sum(losses)/len(losses)}')
    self.model.train()

  def update_lr(self, optimizer, scheduler, epoch=-1):
    """Take a step in the LR scheduler. If the scheduler is StepLR, take only
    a step if epoch > 0. Scheduler can be StepLR or OneCycleLR. Also, call the
    log loss method with epoch as param. """
    self.log_loss(epoch)
    if scheduler is None:
      return
    elif 'OneCycleLR' in str(scheduler):
      scheduler.step()
      logging.info(f'New lr: {optimizer.param_groups[0]["lr"]}')
    elif 'StepLR' in str(scheduler) and epoch > 0:
      scheduler.step()
      logging.info(f'New lr: {optimizer.param_groups[0]["lr"]}')
  
  def log_loss(self, epoch=-1):
    """ If epoch=-1: log avg. loss of the last 100 batches. Before resetting
    the cnt and current_loss, add them to the totals for the epoch.
    Else: epoch has ended - log its avg. loss, set all counters to zero
    and call save_model(). """
    self.epoch_loss += self.current_loss
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
