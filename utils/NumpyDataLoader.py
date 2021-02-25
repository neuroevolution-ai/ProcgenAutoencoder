import numpy as np
from torch.utils.data import Dataset

'''
Class for Loading a .npy File in the right format for pytorch
pytorch is working with Tensors of Shape [NxWxHxC] with:
    N: Batch Size
    W: Width
    H: Height
    C: Number Channels (1 Grayscale, 3 RGB)

Numpy Saves the files as [NxCxWxH] therefore a permutation is needed
'''


class NumpyDataLoader(Dataset):
    def __init__(self, path):
        self.datas = np.load(path)
        self.datas = np.transpose(self.datas, (0, 3, 2, 1))
        self.length = self.datas.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.datas[idx]
