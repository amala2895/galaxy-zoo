import torch
import pandas as pd
import csv
import numpy as np
import os
import zipfile

training_csv = 'data/training_solutions_rev1.csv'
training_csv_zip='data/training_solutions_rev1.zip'


def ylabel_raw():
    label_values_file = 'data/label_values.npy'
  
    label_ids_file = 'data/label_ids.npy'
    if os.path.exists(label_values_file) and  os.path.exists(label_ids_file):
        
        label_ids = np.load(label_ids_file)
        label_values = np.load(label_values_file).item()
        
        return label_ids, label_values
    if not os.path.exists(training_csv):
        if not os.path.exists(training_csv_zip):
            raise(RuntimeError("Could not find " + training_csv_zip))
        else:
            zip_ref = zipfile.ZipFile(training_csv_zip, 'r')
            #print("unzipped")
            zip_ref.extractall('data/')
            print("unzipped")
            zip_ref.close()

    label_values = {}
    label_ids=[]
    reader = csv.reader(open(training_csv, 'r'))
    next(reader)
    for row in reader:
        k=row[0]
        k=int(k)
        label_ids.append(k)
        
    len(label_ids)
    reader = csv.reader(open(training_csv, 'r'))
    next(reader)
    for row in reader:
        imageID,v =row[0],row[1:]
        imageID=int(imageID)
        newRow = []
        
        for e in v:
            newRow.append(float(e))
            label_values[imageID] = newRow
    np.save(label_values_file, label_values)
    np.save(label_ids_file, label_ids)
   
    return label_ids, label_values

def getYlabel(no_train, no_val):
    
    label_ids, label_values=ylabel_raw()
    total=len(label_ids)
    
    if not no_train+no_val<total:
        raise(RuntimeError("Total plus validation should be less than "+str(total)))
        
    label_ids_training=label_ids[0:no_train]
    label_ids_validation=label_ids[no_train:no_train+no_val]
    
    
    label_values_training = {}
    label_values_validation = {}
    for imageID in label_ids_training:
        label_values_training[imageID]=label_values[imageID]
        
    for imageID in label_ids_validation:
        label_values_validation[imageID]=label_values[imageID]
    
    return label_ids_training, label_ids_validation, label_values_training, label_values_validation
    
    
    
    
    
    
    
    
    
    
    
    
       