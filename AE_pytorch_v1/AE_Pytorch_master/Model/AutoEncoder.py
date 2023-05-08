import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder分支网络_1 
        self.conv1_1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1_2 = nn.Conv2d(64,32,kernel_size=5,stride=1,padding=2)
        self.bn1_2 = nn.BatchNorm2d(32)

        #encoder分支网络_2
        self.conv2_1 = nn.Conv2d(1,64,kernel_size=5,stride=2,padding=2)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64,32,kernel_size=5,stride=2,padding=2)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(65536,64)#待定
        
    def forward(self,input):
        out1 = self.conv1_1(input) #[4,64,128,128]   
        out1 = self.bn1_1(out1) #[4,64,128,128]
        out1 = self.pooling(out1) #[4,64,64,64]
        out1 = self.conv1_2(out1) #[4,32,62,62]
        out1 = self.bn1_2(out1) #[4,32,64,64]
        out1 = self.pooling(out1) #[4,32,32,32]

        out2 = self.conv2_1(input) #[4,64,64,64]
        out2 = self.bn2_1(out2) #[4,64,64,64]
        out2 = self.conv2_2(out2) #[4,32,32,32]
        out2 = self.bn2_2(out2) #[4,32,32,32]

        out = torch.cat((out1,out2),dim=1)#[4,64,32,32]
        out = self.flatten(out) #65536
        out = self.linear(out) 
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64,65536)#待定
        self.unflatten = nn.Unflatten(1,[64,32,32])
        
        #decoder网络分支_1
        self.tconv1_1 =nn.ConvTranspose2d(64, 32, kernel_size=3,stride=1,padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.tconv1_2 =nn.ConvTranspose2d(32, 64, kernel_size=3,stride=1,padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        #decoder网络分支_2
        self.tconv2_1 =nn.ConvTranspose2d(64, 32, kernel_size=2,stride=2)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.tconv2_2 =nn.ConvTranspose2d(32, 64, kernel_size=2,stride=2)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(128,1,kernel_size=1,stride=1)
        # self.bn = nn.BatchNorm2d(1)

    def forward(self,input):
        input = self.linear(input)
        input = self.unflatten(input)#[4,64,32,32]

        out1 = self.tconv1_1(input)#[4,32,32,32]
        out1 = self.bn1_1(out1)#[4,32,32,32]
        out1 = self.upsampling(out1)#[4,32,64,64]
        out1 = self.tconv1_2(out1)#[4,64,64,64]
        out1 = self.bn1_2(out1)#[4,64,64,64]
        out1 = self.upsampling(out1)#[4,64,128,128]

        out2 = self.tconv2_1(input)#[4,32,64,64]
        out2 = self.bn2_1(out2)#[4,32,64,64]
        out2 = self.tconv2_2(out2)#[4,64,128,128]
        out2 = self.bn2_2(out2)#[4,64,128,128]

        out = torch.cat((out1,out2),dim=1)
        out = self.conv1(out)
        # nn.GELU

        return out

# test=torch.zeros([4,1,128,128])
# output=Encoder()(test)

# test=torch.zeros([4,64])
# output=Decoder()(test)
class Autoencoder(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,input):
        out=self.encoder(input)
        out=self.decoder(out)
        return out
