#converting traning_solutions_rev1.csv file to y labels ids and values
import torch
import pandas as pd
import csv
import numpy as np
import os
import zipfile

training_csv = 'data/training_solutions_rev1.csv'
training_csv_zip='data/training_solutions_rev1.zip'

def separate():
    
    label_values_training_file = 'data/label_values_training.npy'
    label_values_validation_file = 'data/label_values_validation.npy'
    label_ids_training_file = 'data/label_ids_training.npy'
    label_ids_validation_file = 'data/label_ids_validation.npy'
    
    if os.path.exists(label_values_training_file) and os.path.exists(label_values_validation_file) and os.path.exists(label_ids_training_file) and os.path.exists(label_ids_validation_file):
        label_ids_training = np.load('data/label_ids_training.npy')
        label_values_training = np.load('data/label_values_training.npy').item()
        label_ids_validation = np.load('data/label_ids_validation.npy')
        label_values_validation = np.load('data/label_values_validation.npy').item()
        return label_ids_training, label_ids_validation, label_values_training, label_values_validation
    
    if not os.path.exists(training_csv):
        if not os.path.exists(training_csv_zip):
            raise(RuntimeError("Could not find " + 'data/training_solutions_rev1.csv'))
        else:
            zip_ref = zipfile.ZipFile(training_csv_zip, 'r')
            #print("unzipped")
            zip_ref.extractall('data/')
            print("unzipped")
            zip_ref.close()

    label_values_training = {}
    label_values_validation = {}
    
              
   
              
    reader = csv.reader(open(training_csv, 'r'))

    label_ids=[]
    next(reader)
    for row in reader:
        k=row[0]
        k=int(k)
        label_ids.append(k)

    len(label_ids)

    label_ids_training=label_ids[0:49001]
    label_ids_validation=label_ids[49001:]
    #print(len(label_ids_training))
    #print(len(label_ids_validation))
    #print(len(label_ids_training)+len(label_ids_validation))
    
        
    
    reader = csv.reader(open(training_csv, 'r'))
    next(reader)
    for row in reader:
        imageID,v =row[0],row[1:]
        imageID=int(imageID)
        newRow = []
        if(imageID in label_ids_training):
            #print(k)
            for e in v:
                newRow.append(float(e))
            label_values_training[imageID] = newRow
        else:
            for e in v:
                newRow.append(float(e))
            label_values_validation[imageID] = newRow


    print("Length of training set after dividing: " + str(len(label_values_training)))
    print("Length of training set after dividing: " + str(len(label_values_validation)))


    np.save('data/label_values_training.npy', label_values_training)
    np.save('data/label_values_validation.npy', label_values_validation)

    np.save('data/label_ids_training.npy', label_ids_training)
    np.save('data/label_ids_validation.npy', label_ids_validation)

    print("CSV file divided into training (label_ids_training.npy havinf 49001 data points)" + 
          " and validation (label_ids_validation.npy having 12577 data points)")
              
    return label_ids_training, label_ids_validation, label_values_training, label_values_validation