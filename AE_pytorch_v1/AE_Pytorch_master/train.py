import torch
import torch.nn as nn
import torch.optim as optim
from Model.AutoEncoder import *
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import *
import random
import numpy as np
from torchvision.utils import save_image
from config import params
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 随机数种子
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)


#设置GPU
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

#加载数据
ERA5_data = ERA5_dataset(params['file_path'],transform=transform)
my_dataloader = DataLoader(ERA5_data,params['batch_size'],shuffle=False)
#初始化网络
encoder =Encoder()
decoder =Decoder()
model = Autoencoder(encoder,decoder).to(device)
#初始化参数

#损失函数
criterion = nn.MSELoss()

#优化函数
optimizer = optim.Adam(model.parameters(), params['learning_rate'])

# 初始化学习率调度器
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_history = []

#开始训练
print("-"*25)
print("Starting Training Loop...")
print("-"*25)
for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for data in tqdm(my_dataloader):
        img_batch = data['image'].to(device)
        # ===================forward=====================
        output = model(img_batch)
        loss = criterion(img_batch,output)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     # ===================log========================
    epoch_time = time.time() - epoch_start_time
    print('epoch [{}/{}], loss:{:.4f},time: {:.2f}s'
          .format(epoch + 1, params['num_epochs'], loss.item(),epoch_time))
    loss_history.append(loss.item())

    # 更新学习率
    scheduler.step(loss)

        
    # 保存网络权重、解码器图像
    if (epoch+1) % params['save_epoch'] == 0:
        pic = output.cpu().data[0]
        save_image(pic, './AE_Pytorch_master/results/image_{}.png'.format(epoch))

        torch.save(encoder.state_dict(), './AE_Pytorch_master/pth/encoder_epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), './AE_Pytorch_master/pth/epoch_{}.pth'.format(epoch))
        


# # 保存最后网络权重
# torch.save(model.state_dict(), './AE_Pytorch_master/pth/final_epoch.pth')

#训练损失
plt.figure(figsize=(10,5))
plt.title("AutoEncoder Loss During Training")
plt.plot(loss_history, label='Training Loss')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./AE_Pytorch_master/results/loss_history.jpg')