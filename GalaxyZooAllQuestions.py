
# coding: utf-8

# ## Galaxy Zoo All Questions
# 
# ### Python files required to run ths notebook: data_loader.py, separate_training_validation.py

# In[14]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os


# In[16]:


# Training settings
parser = argparse.ArgumentParser(description='Galaxy zoo project')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located")
parser.add_argument('--crop_size', type=str, default=256, metavar='D',
                    help="Crop Size of images")
parser.add_argument('--resolution', type=str, default=64, metavar='D',
                    help="Final Resolution of images")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--question', type=int, default=0, metavar='N',
                    help='Question number for which training has to be done. If 0 then all questions')

parser.add_argument('--model_directory', type=str, default="models", metavar='N',
                    help='directory to store models')

parser.add_argument('--validation_length', type=int, default=20, metavar='N',
                    help='length of valiudation set')

parser.add_argument('--train_length', type=int, default=1000, metavar='N',
                    help='length of train set')

input_args = ""
args = parser.parse_args(input_args)
torch.manual_seed(args.seed)


# In[17]:


### Data Initialization and Loading
from data_loader import initialize_data, loader
initialize_data(args.data) 


# In[18]:


from YLabelCreate import getYlabel

label_ids_training, label_ids_validation, label_values_training, label_values_validation = getYlabel(5000,100)


# In[19]:


crop_size = args.crop_size
resolution = args.resolution
batch_size = args.batch_size

questions = args.question
shuffle=True

transformations = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


# In[20]:


train_loader = loader(label_ids_training, label_values_training, crop_size, resolution, batch_size, shuffle, questions)
shuffle=False
val_loader=loader(label_ids_validation, label_values_validation, crop_size, resolution, batch_size, shuffle, questions)


# In[21]:


from Model_All_Questions import Net


# In[22]:


model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_train=nn.MSELoss()
loss_val=nn.MSELoss(reduction='sum')


# In[23]:


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_train(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                           epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.data[0]))



# In[24]:


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target).float()
        output = model(data)
        validation_loss += loss_val(output, target) # sum up batch loss
        
    validation_loss /= len(val_loader.dataset)
    


    print('\nValidation set: Average loss: {:.4f}\n'.format(validation_loss, correct))
        
            


# In[25]:



if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
    
for epoch in range(1, args.epochs + 1):
    train(epoch)
   
    validation()
    
    model_file = args.model_directory+'/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)

