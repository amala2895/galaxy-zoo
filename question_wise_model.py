import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

possible_answers = [3,2,2,2,4,2,3,7,3,3,6] #  Number of answers per question


class Net(nn.Module):
    def __init__(self, question_number):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        
        self.conv2 = nn.Conv2d(64, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv2_drop = nn.Dropout2d(p=0.15)
        
        #self.conv3 = nn.Conv2d(32, 16, kernel_size=3)
        #self.bn3 = nn.BatchNorm2d(16)
        #self.conv3_drop = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(29*29*8, 32)
        self.fc1_drop = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(32, possible_answers[question_number-1])

    def forward(self, x):
        x = self.bn1(self.conv1_drop(F.relu(self.conv1(x))))
        x = F.max_pool2d(self.bn2(self.conv2_drop(F.relu(self.conv2(x)))), 2)
        #x = F.max_pool2d(self.bn3(self.conv3_drop(F.relu(self.conv3(x)))), 2)

        x = x.view(-1, 29*29*8)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        
        x = F.relu(self.fc2(x))
        #x = x/(np.sum(x) + 1e-6)
        #print(x)
        return x
