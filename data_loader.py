from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])


def initialize_data(folder):
    train_zip = folder + '/images_training_rev1.zip'
    print(train_zip)
    if not os.path.exists(train_zip):
        raise(RuntimeError("Could not find " + train_zip))
    # extract train_data.zip to train_data
    train_folder = folder + '/images_training_rev1'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # make validation_data by using images 49001 - last
    val_folder = folder + '/images_validation_rev1'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for i,dirs in enumerate(os.listdir(train_folder)):
            #if(i % 100 == 0):
            #    print(i)
            if i > 49000:
                # move file to validation folder
                os.rename(train_folder + '/' + dirs, val_folder + '/' + dirs)
