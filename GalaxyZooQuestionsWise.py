
# coding: utf-8

# ## Galaxy Zoo All Questions
# 
# ### Python files required to run ths notebook: data_loader.py, separate_training_validation.py

# In[23]:


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


# In[24]:


# Training settings
parser = argparse.ArgumentParser(description='Galaxy zoo project')

parser.add_argument('--data', type=str, default='data', metavar='D',
                    help='folder where data is located')

parser.add_argument('--crop_size', type=str, default=256, metavar='D',
                    help='Crop Size of images')

parser.add_argument('--resolution', type=str, default=64, metavar='D',
                    help='Final Resolution of images')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')

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

parser.add_argument('--model_directory', type=str, default='models', metavar='N',
                    help='directory to store models')

parser.add_argument('--validation_length', type=int, default=20, metavar='N',
                    help='length of valiudation set')

parser.add_argument('--train_length', type=int, default=1000, metavar='N',
                    help='length of train set')

parser.add_argument('--outputfile', type=str, default='output.txt', metavar='N',
                    help='outputfile name')

parser.add_argument('--augmentation', type=str, default="", metavar='N',
                    help='augmentation file name')

parser.add_argument('--previous_model', type=str, default='', metavar='N',
                    help='previous model path')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("cuda available")
    print(torch.cuda.device_count())

# In[25]:

outputfile=args.outputfile
### Data Initialization and Loading

from data_loader import initialize_data, loader
initialize_data(args.data) 

ylabelcreate=args.augmentation
x=True
if(ylabelcreate==""):
    x=False
if(x):
    print("doing augmentation")
# In[26]:


from YLabelCreate_augmented import getYlabel

label_ids_training, label_ids_validation, label_values_training, label_values_validation = getYlabel(args.train_length,args.validation_length,x)

import question_map
remove_dependencies =   question_map.getDependencyMap()
q_a=question_map.getQuestionAnswerMap()
for id in label_ids_training:
        
    for i in range(1,12):
        start=q_a[i][0]
        end=q_a[i][1]
        values=remove_dependencies[i]
            
        if values[0]!=-1:
            for v in values:
                m=label_values_training[id][v]
                if(m!=0):
                
                    for j in range(start, end):
                        label_values_training[id][j]=label_values_training[id][j]/m
                        if(label_values_training[id][j]!=1 or label_values_training[id][j]!=0):
                           label_values_training[id][j]=round(label_values_training[id][j],6)
#print(label_values_training)
for id in label_ids_validation:
    
    for i in range(1,12):
        start=q_a[i][0]
        end=q_a[i][1]
        values=remove_dependencies[i]
        
        if values[0]!=-1:
            for v in values:
                m=label_values_validation[id][v]
                if(m!=0):
                    
                    for j in range(start, end):
                        label_values_validation[id][j]=label_values_validation[id][j]/m
                        if(label_values_validation[id][j]!=1 or label_values_validation[id][j]!=0):
                           label_values_validation[id][j]=round(label_values_validation[id][j],6)



# In[27]:


crop_size = args.crop_size
resolution = args.resolution
batch_size = args.batch_size

questions = args.question
shuffle=True

transformations_train = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.Resize(resolution),

                transforms.ToTensor()
    ])
transformations_val = transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.Resize(resolution),

                    transforms.ToTensor()
                                            ])
# In[28]:


train_loader = loader(label_ids_training, label_values_training, crop_size, resolution, batch_size, shuffle, questions,transformations_train)
shuffle=False
val_loader=loader(label_ids_validation, label_values_validation, crop_size, resolution, batch_size, shuffle, questions,transformations_val)


# In[29]:


from question_wise_model import Net


# In[30]:


model = Net()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_train=nn.MSELoss()
loss_val=nn.MSELoss(reduction='sum')


# In[31]:

if torch.cuda.is_available():
    model.cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data=data.cuda()
            target=target.cuda()
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_train(output, target)
        loss = Variable(loss, requires_grad = True)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            f=open(outputfile, 'a')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                           epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item()))
            f.write('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                           epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item()))
            f.close()

# In[32]:


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if torch.cuda.is_available():
            data=data.cuda()
            target=target.cuda()
        data, target = Variable(data, volatile=True), Variable(target).float()
        output = model(data).float()
        loss=loss_val(output, target)
        loss = Variable(loss, requires_grad = True)
        validation_loss += loss # sum up batch loss
        
    validation_loss /= len(val_loader.dataset)
    


    print('\nValidation set: Average loss: {:.4f}\n'.format(validation_loss, correct))

    f=open(outputfile, 'a')

    f.write('\nValidation set: Average loss: {:.4f}\n'.format(validation_loss, correct))
    f.close()

# In[ ]:



if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
    
for epoch in range(1, args.epochs + 1):
    train(epoch)
   
    validation()
    if(epoch%1==0):
        model_file = args.model_directory+'/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)

model_file = args.model_directory+'/model_' + str(epoch) + '.pth'
torch.save(model.state_dict(), model_file)
