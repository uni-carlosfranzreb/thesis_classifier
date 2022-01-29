""" Load the documents, and retrieve their respective word vectors from the
fasttext file. This class is an argument for PyTorch's DataLoader."""



from torch.utils.data import IterableDataset
from torch import LongTensor


class Dataset(IterableDataset):
  def __init__(self, folder):
    """ Initializes the Dataset.
    folder (str): location of the folder that contains the data files. The
    file structure is explained in the README. """
    pass