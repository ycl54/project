import h5py
import os
from PIL import Image
import numpy as np
data_dir = "./mvtec/bottle"
train_data = []
valid_data = []
for i in range(len(os.listdir(data_dir+'/train/good'))):
    if(i < 10):
        im = Image.open(data_dir+'/train/good/00{}.png'.format(i))
    elif(i < 100):
        im = Image.open(data_dir+'/train/good/0{}.png'.format(i))
    else:
        im = Image.open(data_dir+'/train/good/{}.png'.format(i))

    train_data.append(np.uint8(im))
    valid_data.append(np.uint8(im))

file = h5py.File('dataset.h5', 'w')

file.create_dataset('train', data=np.array(train_data))






