from __future__ import print_function
import zipfile
import os
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import torch
import numpy as np



import torchvision.transforms as transforms

image_folder = 'data/images_training_rev1'

#3,2,2,2,4,2,3,7,3,3,6   Number of answers per question
question_starts = [0,3,5,7,9,13,15,18,25,28,31,37]


# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

#data_transforms = transforms.Compose([                    UNCOMMENT IF TRANSFORMS WORK CORRECTLY
#    transforms.Resize((128, 128)),
#    transforms.ToTensor(),
#    #transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
#])

def initialize_test_data(folder):
    test_zip = folder + '/images_test_rev1.zip'
   
    if not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + test_zip))
    
    test_folder = folder + '/images_test_rev1'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
        
def initialize_data(folder):
    train_zip = folder + '/images_training_rev1.zip'
   
    if not os.path.exists(train_zip):
        raise(RuntimeError("Could not find " + train_zip))
    # extract train_data.zip to train_data
    train_folder = folder + '/images_training_rev1'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
        
    
        
    # make validation_data by using images 49001 - last           UNCOMMENT ALL IF DATA NEEDS TO BE DIVIDED IN TRAINING AND VALIDATION
    #val_folder = folder + '/images_validation_rev1'
    #if not os.path.isdir(val_folder):
    #    print(val_folder + ' not found, making a validation set')
    #    os.mkdir(val_folder)
    #    for i,dirs in enumerate(os.listdir(train_folder)):
    #        #if(i % 100 == 0):
    #        #    print(i)
    #        if i > 49000:
    #            # move file to validation folder
    #            os.rename(train_folder + '/' + dirs, val_folder + '/' + dirs)
    
    
def loader(label_ids, label_values, crop_size, resolution, batch_size, shuffle, questions,transforms1=None):
    
    class Dataset(data.Dataset):
      #'Characterizes a dataset for PyTorch'
        def __init__(self, list_IDs, labels, image_folder, crop_size, resolution, questions):
            'Initialization'
            self.labels = labels
            self.list_IDs = list_IDs
            self.transforms=transforms1
            self.image_folder=image_folder
            self.crop_size = crop_size
            self.resolution = resolution
            self.questions = questions

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]

            # Load data and get label
            img = Image.open(self.image_folder + '/' + str(ID) + '.jpg')
            
            #to do self transforms 
            X= transforms.CenterCrop(self.crop_size)(img)
            X= transforms.Resize(self.resolution)(X)
            X = transforms.ToTensor()(X)
            if self.transforms is not None:
                X = self.transforms(X)
                
            y = torch.from_numpy(np.array(self.labels[ID]))
            if(self.questions != 0):
                y = y[question_starts[self.questions-1] : question_starts[self.questions]]
                #y = y[0:2]
            return X, y
        
    if(questions > 11 or questions < 0):
        raise(RuntimeError("Incorrect question number! Valid range: 0 - 11"))
    
    data_set = Dataset(label_ids, label_values, image_folder, crop_size, resolution, questions)
    
  
    
    params = {'batch_size': batch_size,
              'shuffle': shuffle}
    data_loader= DataLoader(data_set, **params)
   
    
    return data_loader

