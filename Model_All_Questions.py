import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import question_map

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)#input - 64*64 - output -60*60
        self.pool1 = nn.MaxPool2d(2, 2)#input - 60*60 - output -30*30
        self.conv2 = nn.Conv2d(16, 128, 5)#input - 30*30 - output -26*26
        self.pool2 = nn.MaxPool2d(2, 2)#input - 26*26 - output -13*13
        self.fc1 = nn.Linear(128 * 13 * 13, 256)
        self.fc2 = nn.Linear(256, 37)
        self.remove_dependencies = question_map.getDependencyMap()
        
        
        self.q_a=question_map.getQuestionAnswerMap()
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x=self.pool1(x)
        x = F.relu(self.conv2(x))
        x=self.pool2(x)
        n=self.num_flat_features(x)
        x = x.view(-1, n)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        #print(x[0])
        y=F.sigmoid(x)
        #print(x.shape)
        normalized=self.normalize(y) 
        #print(normalized.shape)
        add_dependencies=self.multiply_prob(normalized)
        
        #print(add_dependencies.shape)
        
        
        return add_dependencies
    
    def normalize(self,x):
        
        tensor_list = []
        for each in x:
            allArrays = np.array([])
            for i in range(1,12):
                start=self.q_a[i][0]
                end=self.q_a[i][1]
                #print(i)
                #print(each[start:end])
                n=torch.div(each[start:end],torch.sum(each[start:end])+1e-12).detach().numpy()
                #print(n)
                allArrays = np.concatenate([allArrays, n])
            newtorch=torch.from_numpy(allArrays)
          
            tensor_list.append(newtorch)
          
       
        stacked_tensor = torch.stack(tensor_list)
       
        return stacked_tensor
    
    def multiply_prob(self,n):
        tensor_list = []
        for each in n:
            #print(each.shape)
            allArrays = np.array([])
            for i in range(1,12):
                #print(i)
                start=self.q_a[i][0]
                end=self.q_a[i][1]
                arr=each[start:end].detach().numpy()
                #print(arr)
                values=self.remove_dependencies[i]
                if values[0]!=-1:
                    for v in values:
                        m=each[v].detach().numpy()
                        #print(m)
                        arr=np.multiply(arr,m)
                #print(arr)
                
                allArrays = np.concatenate([allArrays, arr])
            #print(allArrays.shape)
            
            
            newtorch=torch.from_numpy(allArrays)
          
            tensor_list.append(newtorch)
          
       
        stacked_tensor = torch.stack(tensor_list)
       
        return stacked_tensor
            
        
        
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features