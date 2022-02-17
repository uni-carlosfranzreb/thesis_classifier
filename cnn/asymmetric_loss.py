""" From https://github.com/Alibaba-MIIL/ASL """


import torch
from torch import nn

class ASL(nn.Module):
	def __init__(self, gamma_neg=4, gamma_pos=0):
		super(ASL, self).__init__()
		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos

	def forward(self, x, y):
		loss_pos = y * torch.pow(1-x, self.gamma_pos) * torch.log(x)
		loss_neg = (1-y) * torch.pow(x, self.gamma_neg) * torch.log(1-x)
		return -torch.mean(loss_pos + loss_neg)