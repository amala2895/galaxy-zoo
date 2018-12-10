import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)#input - 64*64 - output -60*60
        self.pool1 = nn.MaxPool2d(2, 2)#input - 60*60 - output -30*30
        self.conv2 = nn.Conv2d(16, 128, 5)#input - 30*30 - output -26*26
        self.pool2 = nn.MaxPool2d(2, 2)#input - 26*26 - output -13*13
        self.fc1 = nn.Linear(128 * 13 * 13, 256)
        self.fc2 = nn.Linear(256, 37)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x=self.pool1(x)
        x = F.relu(self.conv2(x))
        x=self.pool2(x)
        n=self.num_flat_features(x)
        x = x.view(-1, n)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features