""" Maximum-constraint loss. """


import torch
from torch import nn

class MCL(nn.Module):
	def __init__(self, mask, gamma_neg=0, gamma_pos=0):
		""" The mask is the same mask that is input in the coherent model.
		The gammas are the same ones as in ASL. """
		super(MCL, self).__init__()
		self.mask = mask
		self.ones = torch.ones(mask.shape[0])
		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos


	def forward(self, x, y):
		""" After preparing the input, compute ASL. """
		x = self.prepare_input(x, y)
		loss_pos = y * torch.pow(1-x, self.gamma_pos) * torch.log(x)
		loss_neg = (1-y) * torch.pow(x, self.gamma_neg) * torch.log(1-x)
		return -torch.mean(loss_pos + loss_neg)
	
	def prepare_input(self, x, y):
		""" Prepare the input as explained in the paper. """
		masked = x.mul(self.mask.unsqueeze(1)).transpose(0, 1)
		out_y = x.mul(y).unsqueeze(2).transpose(1, 2)
		max_masked_outy = masked.mul(out_y).max(dim=2).values
		return (self.ones-y).mul(x) + y.mul(max_masked_outy)
