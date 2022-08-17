"""This module contains data read/save functions """
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from osgeo import gdal
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt  

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

class MyDataset(Dataset):
    def __init__(self, args, A_path, B_path, lab_path):
        super(MyDataset, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(A_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist_lab = [name for name in os.listdir(lab_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        self.A_filenames = [os.path.join(A_path, x) for x in datalist if is_image_file(x)]
        self.B_filenames = [os.path.join(B_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [os.path.join(lab_path, x) for x in datalist_lab if is_image_file(x)]  

        self.transform_RGB_A = get_transform(convert=True, normalize=True, is_pre=True, isRGB=True) 
        self.transform_RGB_B = get_transform(convert=True, normalize=True, is_pre=False, isRGB=True) 

        self.transform_label = get_transform() 
        self.out_cls = args.out_cls


    def __getitem__(self,index):
        fn = self.A_filenames[index]
        A_img = self.transform_RGB_A(Image.open(self.A_filenames[index]).convert('RGB')) 
        B_img = self.transform_RGB_B(Image.open(self.B_filenames[index]).convert('RGB'))

        label = self.transform_label(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), self.out_cls).squeeze(0)  


        return A_img, B_img, label , index

    def __len__(self):
        return len(self.A_filenames)


def get_transform(convert=True, normalize=False, is_pre=True, isRGB=True):

    transform_list = [] 
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        if is_pre:
            if isRGB:
                transform_list += [transforms.Normalize((0.4355, 0.4382, 0.4335),
                                                        (0.2096, 0.2089, 0.2095))]                                                           
        else:
            if isRGB:
                transform_list += [transforms.Normalize((0.3306, 0.3351, 0.3297), 
                                                        (0.1612, 0.1667, 0.1607))]    
     
    return transforms.Compose(transform_list)

