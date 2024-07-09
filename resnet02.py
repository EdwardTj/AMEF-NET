import math

import  torch
from    torch import  nn
from    torch.nn import functional as F

# ECA
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # print(kernel_size,type(kernel_size))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
            # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size+1

        padding = kernel_size // 2
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1*1*1)
        # self.maxpool = nn.AdaptiveMaxPool3d(output_size=1*1*1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()
        # self.gelu = nn.GELU()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        # print("eca:",inputs.shape)

        b, c, h, w, d = inputs.shape

        # 全局平均池化 [b,c,h,w，d]==>[b,c,1,1,1]
        x = self.avg_pool(inputs)

        # 维度调整，变成序列形式 [b,c,1,1,1]==>[b,c,1]
        x = x.view([b, c,-1])

        # [b,1, c]
        x = torch.permute(x,(0,2,1))
        # print("eca:", x.shape) #([30, 1, 1, 1,32])
        # 1D卷积
        x = self.conv(x)
        # print("eca:", x.shape)
        # 权值归一化
        x = self.sigmoid(x)
        # x=self.gelu(x)
        # print("eca:", x.shape)
        # 维度调整 [b,c,1]
        x = torch.permute(x, (0, 2, 1))
        x = x.view([b, c, 1, 1, 1])
        # print("eca:", x.shape)
        # 将输入特征图和通道权重相乘[b,c,h,w,d]*[b,c,1,1,1]==>[b,c,h,w,d]
        outputs = x * inputs
        # print(outputs.shape)
        return outputs

# 深度可分离卷积
class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=7,
                                    # kernel_size=3,
                                    stride=1,
                                    padding=3,
                                    groups=in_ch)
        self.point_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=(1,1,1)):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(3,3,3), stride=stride, padding=(1,1,1), bias=False),
            nn.BatchNorm3d(ch_out),
            nn.GELU(),
            nn.Conv3d(ch_out, ch_out, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(ch_out)
        )
        self.gelu=nn.GELU()
        self.shortcut = nn.Sequential()
        if stride != (1,1,1) or ch_in != ch_out:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=(1,1,1), stride=stride, bias=False),
                nn.BatchNorm3d(ch_out)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = self.gelu(out)

        return out

class Upsample(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in,ch_out):
        super(Upsample, self).__init__()
        # self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=(1,1,1), stride=1, padding=0)
        # self.bn1 = nn.BatchNorm3d(ch_out)
        self.conv2 = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=(2,2,2), stride=(2,2,2))
        self.bn2 = nn.BatchNorm3d(ch_out)
        self.gelu = nn.GELU()

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.gelu(out)

        return out

class changeshapeTox4(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in):
        super(changeshapeTox4, self).__init__()
        # self.conv1 = nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv3d(ch_in, 256, kernel_size=(1,1,1), stride=1, padding=0)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(8,8,8))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(8,8,8))
        self.bn = nn.BatchNorm3d(256)
        self.gelu = nn.GELU()

    def forward(self, x):
        x=self.conv2(x)
        out = self.bn(x)
        out = self.gelu(out)
        x1 = self.maxpool(out)
        x2 = self.avgpool(out)
        out = (x1 + x2) / 2

        return out


class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.ch_in=64
        self.conv1 = nn.Sequential(
            # nn.Conv3d(1, 32, kernel_size=(7,7,7), stride=2, padding=3),
            # nn.BatchNorm3d(32),
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),

            DEPTHWISECONV(32,64),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),

        )

        # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()
        # followed 4 blocks
        # [b, 16, h, w, D] => [b, 32, h ,w, D]
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_class)



        # self.gelu = nn.GELU()
        # 通道数
        # 256 256 256
        # x4上采样成x3,通道数调整为256
        self.upsample4 = Upsample(512,256)
        # x3上采样成x2,通道数调整为128
        self.upsample3 = Upsample(256,128)
        # x2上采样成x1,通道数调整为64
        self.upsample2 = Upsample(128,64)


        self.eca_x4 = eca_block(512)
        self.eca_x3 = eca_block(256)
        self.eca_x2 = eca_block(128)
        self.eca_x1 = eca_block(64)

        self.p1changeshapeTox4 = changeshapeTox4(64)
        self.p2changeshapeTox4 = changeshapeTox4(128)
        self.p3changeshapeTox4 = changeshapeTox4(256)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1,1))
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(1,1,1))

        self.dropout = nn.Dropout(p=0.5,inplace=True)
        self.gelu=nn.GELU()
        self.x4changechannel=nn.Conv3d(512,256,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))
        # self.dropout3d = nn.Dropout3d(p=0.7,inplace=True)
        # [b, 256, 7, 7]
        self.linear = nn.Linear(1024, 512)
        self.linear1 = nn.Linear(512, num_class)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch_in, channels, stride))
            self.ch_in = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # print("x1:",x.shape)
        x=x.to(torch.float32)#(1, 256, 256, 256)
        x = self.conv1(x) #(32, 64, 64, 64)
        # print("x.shape:",x.shape)

        x1 = self.layer1(x) #(64, 64, 64, 64)
        # print("blk1:", x1.shape)
        x2 = self.layer2(x1) #(128, 32, 32, 32)
        # print("blk2:", x2.shape)
        x3 = self.layer3(x2) #(1256, 16, 16, 16)
        # print("blk3:", x3.shape)
        x4 = self.layer4(x3) #(512, 8, 8, 8)
        # print("blk4:", x4.shape)

        # x4,x3,x2上采样为p3,p2,p1
        p3 = self.upsample4(x4) #p3 = x3 (128, 16, 16, 16)
        # print("p3:", p3.shape)
        p2 = self.upsample3(x3) #p2 = x2 (64, 32, 32, 32)
        # print("p2:", p2.shape)
        p1 = self.upsample2(x2) #p1 = x1 (32, 64, 64, 64)
        # print("p1:", p1.shape)

        # x3,x2,x1做eca成 eca_x3,eca_x2,eca_x1,
        x4 = self.eca_x4(x4)
        x4 = self.x4changechannel(x4)
        eca_x3 = self.eca_x3(x3)  # (128, 16, 16, 16)
        # print("eca_x3:", eca_x3.shape)
        eca_x2 = self.eca_x2(x2)  # (64, 32, 32, 32)
        # print("eca_x2:", eca_x2.shape)
        eca_x1 = self.eca_x1(x1)  # (32, 64, 64, 64)
        # print("eca_x1:", eca_x1.shape)

        # x4 ( 256,8,8,8)
        p3 = p3+eca_x3  # (128, 16, 16, 16)
        # print("p3:", p3.shape)
        p2 = p2+eca_x2  # (64, 32, 32, 32)
        # print("p2:", p2.shape)
        p1 = p1+eca_x1  # (32, 64, 64, 64)
        # print("p1:", p1.shape)

        # p1,p2,p3
        p3 = self.p3changeshapeTox4(p3)
        # print("p3:", p3.shape)
        p2 = self.p2changeshapeTox4(p2)
        # print("p2:", p2.shape)
        p1 = self.p1changeshapeTox4(p1)
        # print("p1:", p1.shape)

        # x4,p3,p2,p1在c维度上concat   (1024,8,8,8)
        out = torch.cat((x4,p3,p2,p1),dim=1)
        # print("out.shape:",out.shape)

        # 1024*8*8*8 -->1024*1*1*1
        out1 = self.avgpool(out)
        out2 = self.maxpool(out)
        out = (out1+out2)/2
        # out = self.dropout3d(out)
        out = out.view(x.size(0), -1)
        # print("view.size:",out.shape)
        # out = self.gelu(out)
        # out = self.dropout(out)

        # 1024 -->512
        out = self.linear(out)
        out = self.gelu(out)
        out = self.dropout(out)
        # 512 -->num_class
        out = self.linear1(out)

        # print(out)
        return out



def main():

    model = ResNet18(2)
    # print(model)
    tmp = torch.randn(1, 1,256,256,256)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()