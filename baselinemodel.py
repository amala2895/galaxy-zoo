import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import question_map


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #input - 64*64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        # output 60*60
  
        # input 60*60
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=0.15)
        # output 56*56
        # maxpool
        # input 28*28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_drop = nn.Dropout2d(p=0.2)
        # output 26*26
        # maxpool
        # input 13*13
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4_drop = nn.Dropout2d(p=0.25)
        # output 11*11
        # maxpool
        # 5*5*128
        self.fc1 = nn.Linear(3200,1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(1024,512)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(512,37)
            
        self.remove_dependencies =   question_map.getDependencyMap()
        self.q_a=question_map.getQuestionAnswerMap()



    def forward(self, x):
        
        x = self.bn1(self.conv1_drop(F.relu(self.conv1(x))))
        
        x = F.max_pool2d(self.bn2(self.conv2_drop(F.relu(self.conv2(x)))), 2)
        x = F.max_pool2d(self.bn3(self.conv3_drop(F.relu(self.conv3(x)))), 2)
        x = F.max_pool2d(self.bn4(self.conv4_drop(F.relu(self.conv4(x)))), 2)
        x = x.view(-1, 3200)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        
        
        x = F.sigmoid(self.fc3(x))
        #print(x)
        normalized=self.normalize(x)
 #print(normalized)
        add_dependencies=self.multiply_prob(normalized)
        #print(add_dependencies)
        return add_dependencies


    def normalize(self,x):
        for each in x:
            for i in range(1,12):
                start=self.q_a[i][0]
                end=self.q_a[i][1]
                for j in range(start,end):
                    each[j]=torch.div(each[j],torch.sum(each[start:end])+1e-12)
        return x


    def multiply_prob(self,n):
        for each in n:
            for i in range(1,12):
                start=self.q_a[i][0]
                end=self.q_a[i][1]
                values=self.remove_dependencies[i]
                if values[0]!=-1:
                    for v in values:
                        m=each[v]
                        for j in range(start, end):
                            each[j]=each[j]*m
        return n


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
