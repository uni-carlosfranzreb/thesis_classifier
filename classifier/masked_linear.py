""" A masked version of the fully connected linear layer. The mask is a binary
matrix, with ones for the connections that should be kept and else zeros.
From https://github.com/karpathy/pytorch-made/blob/master/made.py """


from torch.nn import Linear
from torch.nn import functional as F


class MaskedLinear(Linear):
  """ same as Linear except has a configurable mask on the weights """
  def __init__(self, mask, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)        
    self.register_buffer('mask', mask.T)

  def forward(self, input):
    return F.linear(input, self.mask * self.weight, self.bias)