from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

import torchvision.transforms as transforms
from data_loader import initialize_test_data
from Model_All_Questions import Net

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--model', type=str, metavar='M',default='models/model_5.pth',
                    help="the path to model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='submission.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--data_directory', type=str, default='data', metavar='D',
                    help="data directory")
parser.add_argument('--crop_size', type=str, default=256, metavar='D',
                    help="Crop Size of images")
parser.add_argument('--resolution', type=str, default=64, metavar='D',
                    help="Final Resolution of images")


args = parser.parse_args()

initialize_test_data(args.data_directory) 

state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()

test_dir=args.data_directory+'/images_test_rev1'

output_file = open(args.outfile, "w")
output_file.write("GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n")
i=0
for f in tqdm(os.listdir(test_dir)):
    
    img = Image.open(test_dir + '/' + f)
    X= transforms.CenterCrop(args.crop_size)(img)
    X= transforms.Resize(args.resolution)(X)
    data = transforms.ToTensor()(X)
    data = data.view(1, data.size(0), data.size(1), data.size(2))
       
    data = Variable(data, volatile=True)
    output = model(data)
    output=output.data.numpy()
    output=output[0]

    f=f.split('.')[0]
 
    output_file.write("%s," % (f))
    for o in range(0,37):
        if o==36:
            output_file.write("%f"% output[o])
        else:
            output_file.write("%f,"% output[o])
    i=i+1
    
print(i)
    