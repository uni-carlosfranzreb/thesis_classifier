""" Illustrate how the MCM and preparation for the MCLoss look like.
We use three subjects S1, S2, and S3, where S1 is the ancestor of S2 and S3.
The batch consists of two documents, whose outputs have three probabilities
each, one for each subject. We expect the result to increase the probability
for S1 if the probability for S2 or S3 is higher, so the hierarchy constraints
are preserved.
"""


import torch


n_subjects = 3
batch_size = 2
out = torch.tensor([
  [.2, 0., 1.],  # out for doc 1 (D1)
  [1., 1., .4],  # out for doc 2 (D2)
])
y = torch.tensor([
  [0, 0, 1],  # true assignments of D1
  [1, 0, 0],  # true assignments of D2
])

mask = torch.tensor([
  [1, 1, 1],  # descendants of subject 1 (S1)
  [0, 1, 0],  # descendants of subject 2 (S2)
  [0, 0, 1],  # descendants of subject 3 (S3)
])  # S1 is an ancestor of S2 and S3

expected = torch.tensor([  # after masking
  [
    [.2, 0., 1.],  # candidates for D1-S1
    [0., 0., 0.],  # candidates for D1-S2
    [0., 0., 1.],  # candidates for D1-S3
  ],
  [
    [1., 1., .4],  # candidates for D2-S1
    [0., 1., 0.],  # candidates for D2-S2
    [0., 0., .4],  # candidates for D2-S3
  ],
])
expected_max = torch.tensor([  # after the max operation
  [1., 0., 1.],  # MCM for D1
  [1., 1., .4],  # MCM for D2
])
expected_max_masked_outy = torch.tensor([  # part of the BCE input
  [1., 0., 1.],
  [1., 0., 0.],
])
expected_bce = torch.tensor([  # final BCE input
  [1., 0., 1.],
  [1., 1., .4]
])

masked = out.mul(mask.unsqueeze(1)).transpose(0, 1) # ? out.mul(mask.unsqueeze(0))
assert torch.all(masked == expected)

mcm = masked.max(dim=2).values
assert torch.all(mcm == expected_max)

ones = torch.ones(mcm.shape[1])
out_y = out.mul(y).unsqueeze(2).transpose(1, 2)
max_masked_outy = masked.mul(out_y).max(dim=2).values
assert torch.all(max_masked_outy == expected_max_masked_outy)

bce_x = (ones-y).mul(mcm) + y.mul(max_masked_outy)
assert torch.all(bce_x == expected_bce)
