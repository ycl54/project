from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import numpy as np
import torch
import os
class Data(Dataset):
    def __init__(self, type, data_dir=None, transform=None):
        super(Data, self).__init__()
        self.type = type
        self.transform = transform
        self.data_dir = data_dir
        f = h5py.File('dataset.h5','r')
        self.ds = f['train']

        self.len = self.ds.shape[0]


        if(self.type == 'test'):
            self.len1 = len(os.listdir(self.data_dir + '/test/good'))
            self.len2 = len(os.listdir(self.data_dir + '/test/contamination'))
            self.len3 = len(os.listdir(self.data_dir + '/test/broken_small'))
            self.len4 = len(os.listdir(self.data_dir + '/test/broken_large'))

            self.len = self.len1 + self.len2 +self.len3 +self.len4

    def __len__(self):



        return self.len

    def __getitem__(self, idx):
        if(self.type != "test"):

            im = Image.fromarray(np.uint8(self.ds[idx]))


            if(self.transform is not None):
                image = self.transform(im)


            return image
        else:
            clas = None
            if(idx < self.len1):
                if(idx < 10):
                    im = Image.open(self.data_dir+'/test/good/00{}.png'.format(idx))
                elif(idx < 100):
                    im = Image.open(self.data_dir+'/test/good/0{}.png'.format(idx))
                else:
                    im = Image.open(self.data_dir+"/test/good/{}.png".format(idx))
                clas = 0

            elif(idx < self.len1 + self.len2):
                idx = idx - self.len1
                if(idx < 10):
                    im = Image.open(self.data_dir+'/test/contamination/00{}.png'.format(idx))
                elif(idx < 100):
                    im = Image.open(self.data_dir+'/test/contamination/0{}.png'.format(idx))
                else:
                    im = Image.open(self.data_dir+'/test/contamination/{}.png'.format(idx))
                clas = 1
            elif(idx < self.len1 + self.len2 + self.len3):
                idx = idx - self.len1 - self.len2
                if(idx < 10):
                    im = Image.open(self.data_dir+'/test/broken_small/00{}.png'.format(idx))
                elif(idx < 100):
                    im = Image.open(self.data_dir+'/test/broken_small/0{}.png'.format(idx))
                else:
                    im = Image.open(self.data_dir+'/test/broken_small/{}.png'.format(idx))
                clas = 1
            else:
                idx = idx - self.len1 - self.len2 -self.len3
                if(idx < 10):
                    im = Image.open(self.data_dir+'/test/broken_large/00{}.png'.format(idx))
                elif(idx < 100):
                    im = Image.open(self.data_dir+'/test/broken_large/0{}.png'.format(idx))
                else:
                    im = Image.open(self.data_dir+'/test/broken_large/{}.png'.format(idx))
                clas = 1
            if(self.transform is not None):
                image = self.transform(im)
            return {'image':image,'class':clas}


