import torch
import os
import netCDF4 as nc
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#定义预处理方法，先中心裁剪128，再进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(128),
    transforms.Normalize((0.5,),(0.5,))
])

class ERA5_dataset(Dataset):
    def __init__(self, folder_path,transform=None):
        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        self.length = len(self.files) * 30 # 每个文件包含三十张图像
        self.transform = transform
        
    def __getitem__(self, index):
        file_index = index // 30 # 获取文件的索引
        var_index = index % 30 # 获取变量的索引0~29
        row_index = var_index // 10 #获取行变量的索引的，对应的就是不同时间维度
        col_index = var_index % 10 #获取列变量的索引的，对应的就是不同气压维度
        file_path = os.path.join(self.folder_path, self.files[file_index])
        dataset = nc.Dataset(file_path)
        img_key = list(dataset.variables.keys())[-1] # 从netCDF文件中读取数值维度
        img = dataset.variables[img_key][:].data #取出所有23x24张的图片
        img = img[row_index*7,col_index,::] #根据索引读取一张图片
        img = img.astype(np.float32)
        if self.transform:
            img = self.transform(img)
        sample = {'image': img}
        dataset.close()
        return sample
        
    def __len__(self):
        return self.length




