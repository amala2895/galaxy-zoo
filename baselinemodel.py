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
      
        # output 60*60
  
        # input 60*60
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        
        # output 56*56
        # maxpool
        # input 28*28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        
        # output 24*24
        # maxpool
        # input 12*12*128
        
   
        self.fc1 = nn.Linear(12*12*128,1024)
             
        self.fc3 = nn.Linear(1024,37)
            
        self.remove_dependencies =   question_map.getDependencyMap()
        self.q_a=question_map.getQuestionAnswerMap()



    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    
        x = x.view(-1, 12*12*128)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        

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
