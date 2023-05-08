import torch
import torchvision.transforms as transforms
from Model.AutoEncoder import *
from PIL import Image
import torch.nn.functional as F

# 加载测试集图片
test_image_path = "E:/Data_local/ERA5_IMG/era5.10m_v_component_of_wind.20220618.nc19.jpg"
test_image = Image.open(test_image_path)
if test_image != 'L':
    test_image = test_image.convert('L')

# 转换图像以符合模型输入
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(128),
    transforms.Normalize((0.5,),(0.5,))
])
test_image_tensor = transform(test_image).unsqueeze(0)  # 添加一个维度以匹配模型输入

#计算均方根误差比较图片
def rmse_loss(output, target):
    mse = F.mse_loss(output, target, reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse

# 加载保存的模型权重
encoder =Encoder()
decoder =Decoder()
model = Autoencoder(encoder,decoder)
model.load_state_dict(torch.load("./AE_Pytorch_master/pth/epoch_15.pth"))

# 在模型上进行推断
with torch.no_grad():
    output = model(test_image_tensor)

# 将输入图像和输出图像可视化进行比较
import matplotlib.pyplot as plt

#中心裁剪输入数据
width, height = test_image.size   # Get dimensions
left = (width - 128)/2
top = (height - 128)/2
right = (width + 128)/2
bottom = (height + 128)/2
test_image = test_image.crop((left, top, right, bottom))

plt.subplot(1, 2, 1)
plt.imshow(test_image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output[0].permute(1, 2, 0))
plt.title("Output Image")
plt.axis("off")

plt.show()

# 计算RMSE评价模型生成图片性能
rmse = rmse_loss(output, test_image_tensor)
print("RMSE: {:.4f}".format(rmse.item()))