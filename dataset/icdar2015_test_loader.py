# -*- coding: UTF-8 -*-
# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch

ic15_test_data_dir = './test_img/'

#ic15_test_data_dir = '../OCR_dataset/funsdata/renamed_testing_data/images/'

#ic15_test_data_dir = '../OCR_dataset/SROIE2019/test_2019_img/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]] # 改变图像通道顺序为：RGB
    except Exception as e:
        print img_path
        raise
    return img

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale) #长宽等比例缩放，长边缩放到2240
    return img, scale

class IC15TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240):
        data_dirs = [ic15_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
            
            self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) / part_num
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img, s = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        # 这里的scale是针对ic15的.之后应该考虑对不同的数据集进行不同的预处理操作
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        #w,h, BGR
        return img[:, :, [2, 1, 0]], scaled_img, s
