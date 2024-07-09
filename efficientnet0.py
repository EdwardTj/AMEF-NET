import math

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups: int = 1,
                 activation: nn.Module = nn.Identity()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.conv(x)



        # print("eca:", x.shape)
        # 将输入特征图和通道权重相乘[b,c,h,w,d]*[b,c,1,1,1]==>[b,c,h,w,d]
        outputs = x * inputs
        # print(outputs.shape)
        return outputs
# class SEModule(nn.Module):
#     # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
#     def __init__(self, in_channels, b=1, gama=2):
#         # 继承父类初始化
#         super(SEModule, self).__init__()
#
#         # 根据输入通道数自适应调整卷积核大小
#
#         kernel_size = int(abs((math.log(in_channels, 2) + b) / gama))
#         # print(kernel_size,type(kernel_size))
#         # 如果卷积核大小是奇数，就使用它
#         if kernel_size % 2:
#             kernel_size = kernel_size
#             # 如果卷积核大小是偶数，就把它变成奇数
#         else:
#             kernel_size = kernel_size+1
#
#         padding = kernel_size // 2
#         # 全局平均池化，输出的特征图的宽高=1
#         self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1*1*1)
#         # self.maxpool = nn.AdaptiveMaxPool3d(output_size=1*1*1)
#         # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
#         self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
#                               bias=False, padding=padding)
#         # sigmoid激活函数，权值归一化
#         self.sigmoid = nn.Sigmoid()
#         # self.gelu = nn.GELU()
#
#     # 前向传播
#     def forward(self, inputs):
#         # 获得输入图像的shape
#         # print("eca:",inputs.shape)
#
#         b, c, h, w, d = inputs.shape
#
#         # 全局平均池化 [b,c,h,w，d]==>[b,c,1,1,1]
#         x = self.avg_pool(inputs)
#
#         # 维度调整，变成序列形式 [b,c,1,1,1]==>[b,c,1]
#         x = x.view([b, c,-1])
#
#         # [b,1, c]
#         x = torch.permute(x,(0,2,1))
#         # print("eca:", x.shape) #([30, 1, 1, 1,32])
#         # 1D卷积
#         x = self.conv(x)
#         # print("eca:", x.shape)
#         # 权值归一化
#         x = self.sigmoid(x)
#         # x=self.gelu(x)
#         # print("eca:", x.shape)
#         # 维度调整 [b,c,1]
#         x = torch.permute(x, (0, 2, 1))
#         x = x.view([b, c, 1, 1, 1])
#         print("eca:", x.shape)
#         # 将输入特征图和通道权重相乘[b,c,h,w,d]*[b,c,1,1,1]==>[b,c,h,w,d]
#         outputs = x * inputs
#         # print(outputs.shape)
#         return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self,in_channels,b=1, gama=2):
        super(SpatialGate, self).__init__()
        kernel_size= int(abs((math.log(in_channels, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size+1
        padding = kernel_size // 2
        # print("kernel_size:",kernel_size,padding)
        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm3d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmoid(out)

class SEModule(nn.Module):
    def __init__(self,  in_channels,spatial=True,):
        super(SEModule, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate(in_channels)
        self.width_gate = SpatialGate(in_channels)
        if self.spatial:
            self.spatial_gate = SpatialGate(in_channels)
        self.depth_gate = SpatialGate(in_channels)

    def forward(self, x):
        x_perm1 = x.permute(0, 4, 2, 3, 1).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 4, 2, 3, 1).contiguous()

        x_perm2 = x.permute(0, 3, 4, 1, 2).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 4, 1, 2).contiguous()

        if self.spatial:
            x_perm3 = x.permute(0, 1, 4, 3, 2).contiguous()
            x_out3 = self.spatial_gate(x_perm3)
            x_out3 = x_out3.permute(0, 1, 4, 3, 2).contiguous()
            x_out3 = self.depth_gate(x_out3)
            return (1/4) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/3) * (x_out1 + x_out2)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, channels_factor, downsample: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tmp_channels = self.in_channels * channels_factor
        self.stride = 1 if not downsample else 2
        self.conv = nn.Sequential(
            ConvBNSiLU(in_channels=self.in_channels, out_channels=self.tmp_channels, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.tmp_channels, kernel_size=3, stride=self.stride,
                       padding=1, groups=self.tmp_channels, activation=nn.SiLU(inplace=True)),
            SEModule(in_channels),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                       padding=0),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.in_channels == self.out_channels: return out + x
        return out


class EfficientNetB0(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        CFG = [1, 2, 2, 3, 3, 4, 1]
        self.conv = ConvBNSiLU(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.mbconv1_1 = self._make_layers(CFG[0], in_channels=32, out_channels=16, channels_factor=1, downsample=False)
        self.mbconv6_2 = self._make_layers(CFG[1], in_channels=16, out_channels=24, channels_factor=6, downsample=True)
        self.mbconv6_3 = self._make_layers(CFG[2], in_channels=24, out_channels=40, channels_factor=6, downsample=True)
        self.mbconv6_4 = self._make_layers(CFG[3], in_channels=40, out_channels=80, channels_factor=6, downsample=True)
        self.mbconv6_5 = self._make_layers(CFG[4], in_channels=80, out_channels=112, channels_factor=6,
                                           downsample=False)
        self.mbconv6_6 = self._make_layers(CFG[5], in_channels=112, out_channels=192, channels_factor=6,
                                           downsample=True)
        self.mbconv6_7 = self._make_layers(CFG[6], in_channels=192, out_channels=320, channels_factor=6,
                                           downsample=False)
        self.fc = nn.Sequential(
            ConvBNSiLU(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1280, out_features=512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(2048),
            nn.GroupNorm(64, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=128, out_features=n_classes, bias=True)
        )

    def _make_layers(self, num_layers, in_channels, out_channels, channels_factor, downsample):
        layers = [MBConv(in_channels=in_channels, out_channels=out_channels, channels_factor=channels_factor,
                         downsample=downsample)]
        for _ in range(num_layers - 1):
            layers.append(MBConv(in_channels=out_channels, out_channels=out_channels, channels_factor=channels_factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        # print('# Conv output shape:', x.shape)
        x = self.mbconv1_1(x)
        # print('# MBConv1_1 output shape:', x.shape)
        x = self.mbconv6_2(x)
        # print('# MBConv6_2 output shape:', x.shape)
        x = self.mbconv6_3(x)
        # print('# MBConv6_3 output shape:', x.shape)
        x = self.mbconv6_4(x)
        # print('# MBConv6_4 output shape:', x.shape)
        x = self.mbconv6_5(x)
        # print('# MBConv6_5 output shape:', x.shape)
        x = self.mbconv6_6(x)
        # print('# MBConv6_6 output shape:', x.shape)
        x = self.mbconv6_7(x)
        # print('# MBConv6_7 output shape:', x.shape)
        x = self.fc(x)
        # print('# FC output shape:', x.shape)
        return x
#
# def main():
#     model = EfficientNetB0(1,2)
#     # print(model)
#     tmp = torch.randn(2, 1, 192, 192, 192)
#     out = model(tmp)
#     print('resnet:', out.shape)
#
#     p = sum(map(lambda p: p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
# if __name__ == '__main__':
#     main()

# def main():
#     model = EfficientNetB0(1,2)
#     tmp = torch.randn(2, 1,192,192,192)
#     # out = model(tmp)
#     # print('resnet:', out.shape)
#     # print(model)
#     # input = (1, 182, 218, 182)
#     writer = SummaryWriter("pokemon_MRI/logs_seq",tmp)
#     writer.add_graph(model,tmp)
#     writer.close()
#     # writer = summary(model, input_size=(1, 182, 218, 182))
#
#     p = sum(map(lambda p:p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
# if __name__ == '__main__':
#     main()

import torch
from thop import profile
from thop import clever_format
def main():
    model = EfficientNetB0(1,2)
    # print(model)
    # tmp = torch.randn(2, 1, 192, 192, 192)
    # out = model(tmp)
    # print('resnet:', out.shape)
    input_size = (2, 1, 192, 192, 192)
    input_data = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_data,))
    flops = clever_format(flops, "%.3f")
    print(f"FLOPs: {flops}","parameter:",params)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()
