from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import numpy as np
import torch
class Data(Dataset):
    def __init__(self, type, transform=None):
        super(Data, self).__init__()
        self.type = type
        self.transform = transform

        f = h5py.File('dataset.h5','r')
        self.ds = f['train']

        self.len = self.ds.shape[0]

    def __len__(self):


        return self.len

    def __getitem__(self, idx):

        im = Image.fromarray(np.uint8(self.ds[idx]))


        if(self.transform is not None):
            image = self.transform(im)


        return image