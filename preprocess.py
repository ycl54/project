import h5py
import os
from PIL import Image
import numpy as np
data_dir = "./mvtec/bottle"
train_data = []
valid_data = []
test_data = []
print(len(os.listdir(data_dir+'/train/good')))
print(os.listdir(data_dir+'/train/good'))
for i in range(len(os.listdir(data_dir+'/train/good'))):
    print(i)
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
file.create_dataset('test', data=np.array(test_data))





