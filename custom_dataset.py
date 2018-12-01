from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np


import torch
from torch.utils import data

class Dataset(data.Dataset):
  #'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels,image_folder):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms=transforms
        self.image_folder=image_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img = Image.open(self.image_folder + '/' + str(ID) + '.jpg')
        
        #to do self.tramsforms 
        X= transforms.CenterCrop(224)(img)
        X=transforms.Resize(64)(X)
        X = transforms.ToTensor()(X)
        y = torch.from_numpy(np.array(self.labels[ID]))
       
        return X, y