import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
def transform_image(img,ang_range,shear_range,trans_range):
    
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    pts1 = np.float32([[4,4],[18,4],[4,18]])

    pt1 = 4+shear_range*np.random.uniform()-shear_range/2
    pt2 = 18+shear_range*np.random.uniform()-shear_range/2

  
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    
    return img


def complete_data_augmentation(train_folder,images):

    currentDir = os.listdir(train_folder)
    
    for i,file in enumerate(currentDir):
        
        img = plt.imread(train_folder + file)
        #plt.imshow(img)
        #plt.show()
        img_transformed = transform_image(img,90,2,6)
        name = (train_folder + file).split(".")[0]
        plt.imsave(name + '_0' + '.jpg',img_transformed)
        #plt.imshow(img_transformed)
        #plt.show()
        #print(name)
        img_transformed2 = transform_image(img,60,3,12)
        plt.imsave(name + '_1' + '.jpg',img_transformed2)
        #plt.imshow(img_transformed2)
        #plt.show()     
        
complete_data_augmentation("data_augmented/images_training_rev1/", 1)   
        


