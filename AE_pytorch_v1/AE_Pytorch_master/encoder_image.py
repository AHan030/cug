import torch
import torch.nn as nn
from Model.AutoEncoder import *
from Model.AutoEncoder import Encoder # 自编码器的编码器部分
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

weight_path ="./AE_Pytorch_master/pth/encoder_epoch_0.pth"
image_path = "E:/Data_local/ERA5_IMG/era5.10m_v_component_of_wind.20220618.nc19.jpg"

encoder =Encoder()
decoder =Decoder()
model = Autoencoder(encoder,decoder)

def image_encoder(weight_path,image_path):
    # 加载整个自编码器的权重文件
    state_dict = torch.load(weight_path)

    # # 提取编码器部分的权重
    # encoder_weights = {}
    # for key in state_dict.keys():
    #     if key.startswith("encoder"):
    #         encoder_weights[key] = state_dict[key]
    # new_state_dict = {"encoder." + key: value for key, value in encoder_weights.items()}

    # 加载编码器部分
    encoder = model.encoder
    encoder.load_state_dict(state_dict)
    encoder.train() # 设置BatchNormalization层的状态
    encoder.eval()

    # 加载要比较的图像，并进行预处理
    image = Image.open(image_path)
    if image != 'L':
        image = image.convert('L')

    # 转换图像以符合模型输入
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(128),
        transforms.Normalize((0.5,),(0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加一个维度以匹配模型输入

    # 将图像输入到编码器中，得到向量表示
    with torch.no_grad():
        encoded = encoder(image_tensor)
    # 将向量保存到文件中    
    encoded_file = image_path.replace(".jpg", ".pt")

    encoded_vector = encoded.reshape(-1)  # 展平为一维向量
    torch.save(encoded_vector, 'encoded_vector.pt')  # 保存向量到文件
    return encoded_file

encode_file = image_encoder(weight_path,image_path)
print(encode_file)


# # 计算图像之间的相似度
# mse_list = []
# for i in range(num_images):
#     # 加载第 i 张图像的编码器表示
#     encoded_i = torch.load(f'encoded_{i}.pth')
    
#     # 计算 MSE
#     mse = nn.MSELoss()(encoded, encoded_i)
#     mse_list.append((mse.item(), i))
    
# # 根据 MSE 进行排序，得到相似度最高的图像
# mse_list = sorted(mse_list, key=lambda x: x[0])
# most_similar_image_index = mse_list[0][1]